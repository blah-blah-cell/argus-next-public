const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const http = require('node:http');
const vm = require('node:vm');
const {startDevServer,HOST,PORT,CLIENT_PATH,EVENTS_PATH} = require('../../../scripts/desktop-dev-server.cjs');
const {launchPlan} = require('../../../scripts/dev-desktop.cjs');
const projectRoot = path.resolve(__dirname,'../../..');

function get(url,{method='GET',headers={}}={}) {
  return new Promise((resolve,reject) => {
    const request = http.request(url,{method,headers},response => {
      const chunks=[];
      response.on('data',chunk=>chunks.push(chunk));
      response.on('end',()=>resolve({status:response.statusCode,headers:response.headers,body:Buffer.concat(chunks).toString()}));
    });
    request.on('error',reject); request.end();
  });
}
async function fixture(t) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(),'argus-ui-dev-'));
  t.after(()=>fs.rm(root,{recursive:true,force:true}));
  await fs.mkdir(path.join(root,'tests'));
  await fs.writeFile(path.join(root,'index.html'),'<html><body>real controls</body></html>');
  await fs.writeFile(path.join(root,'styles.css'),'body { color: red; }');
  await fs.writeFile(path.join(root,'app.js'),'window.real = true;');
  await fs.writeFile(path.join(root,'package.json'),'{}');
  await fs.writeFile(path.join(root,'tests','private.js'),'not public');
  const development = await startDevServer({root,port:0,debounceMs:10});
  t.after(()=>development.close());
  return {root,...development};
}
test('native launcher selects only the development override and normal debug shell',async()=>{
  const config = JSON.parse(await fs.readFile(path.join(projectRoot,'desktop/src-tauri/tauri.dev.conf.json'),'utf8'));
  const production = JSON.parse(await fs.readFile(path.join(projectRoot,'desktop/src-tauri/tauri.conf.json'),'utf8'));
  assert.deepEqual(Object.keys(config).sort(),['$schema','build']);
  assert.deepEqual(config.build,{devUrl:`http://${HOST}:${PORT}`});
  assert.equal(production.build.devUrl,undefined);
  assert.equal(production.build.frontendDist,'../ui');
  assert.deepEqual(launchPlan(projectRoot),{
    command:'cargo',args:['tauri','dev','--config','src-tauri/tauri.dev.conf.json'],cwd:path.join(projectRoot,'desktop'),
  });
  const launcher=await fs.readFile(path.join(projectRoot,'scripts/dev-desktop.cjs'),'utf8');
  assert.doesNotMatch(launcher,/--release|ARGUS_MODEL_DIR\s*=|ARGUS_DEV_ROOT\s*=|ARGUS_CAPTURE_ROOT\s*=/);
});
test('loopback server serves authored bytes without changing source or accepting stale query versions',async t=>{
  const development=await fixture(t);
  assert.equal(development.server.address().address,'127.0.0.1');
  const html=await get(development.url+'/');
  assert.equal(html.status,200);
  assert.match(html.body,new RegExp(CLIENT_PATH));
  assert.match(html.body,/real controls/);
  assert.equal(html.headers['cache-control'],'no-store');
  assert.doesNotMatch(await fs.readFile(path.join(development.root,'index.html'),'utf8'),/__argus_dev/);
  assert.equal((await get(development.url+'/styles.css?v=old')).body,'body { color: red; }');
  await fs.writeFile(path.join(development.root,'styles.css'),'body { color: blue; }');
  assert.equal((await get(development.url+'/styles.css?v=old')).body,'body { color: blue; }');
  const head=await get(development.url+'/app.js',{method:'HEAD'});
  assert.equal(head.status,200); assert.equal(head.body,'');
  const partial=await get(development.url+'/app.js',{headers:{range:'bytes=0-5'}});
  assert.equal(partial.status,206); assert.equal(partial.body,'window');
  assert.equal((await get(development.url+'/app.js',{headers:{range:'bytes=900-999'}})).status,416);
});
test('watcher sends a reload event after an authored edit; client reloads and disposes',async t=>{
  const development=await fixture(t);
  const request=http.get(development.url+EVENTS_PATH);
  t.after(()=>request.destroy());
  const response=await new Promise((resolve,reject)=>{request.once('response',resolve);request.once('error',reject);});
  response.setEncoding('utf8');
  const event=new Promise((resolve,reject)=>{
    const timeout=setTimeout(()=>reject(Error('No reload event after file change')),3000);
    response.on('data',chunk=>{if(chunk.includes('event: reload')) {clearTimeout(timeout);resolve(chunk);}});
  });
  await fs.writeFile(path.join(development.root,'app.js'),'window.real = 2;');
  assert.match(await event,/data: changed/);
  let listener,disposed=false,reloads=0,pagehide;
  class Events {
    constructor(url) {assert.equal(url,EVENTS_PATH);}
    addEventListener(event,callback) {assert.equal(event,'reload');listener=callback;}
    close() {disposed=true;}
  }
  const script=await get(development.url+CLIENT_PATH);
  vm.runInNewContext(script.body,{
    EventSource:Events,location:{reload(){reloads++;}},
    addEventListener(event,callback){assert.equal(event,'pagehide');pagehide=callback;},
  });
  listener(); assert.equal(reloads,1); pagehide(); assert.equal(disposed,true);
  request.destroy();
});
test('server rejects cross-origin, rebinding, writes, traversal and tooling exposure',async t=>{
  const development=await fixture(t);
  for (const resource of ['/package.json','/tests/private.js','/%2e%2e/Cargo.toml','/missing.js']) {
    assert.equal((await get(development.url+resource)).status,404,resource);
  }
  assert.equal((await get(development.url+'/app.js',{headers:{host:'attacker.example'}})).status,403);
  assert.equal((await get(development.url+'/app.js',{headers:{origin:'https://attacker.example'}})).status,403);
  assert.equal((await get(development.url+'/app.js',{method:'POST'})).status,405);
  assert.equal((await get(development.url+'/%zz.js')).status,400);
});
test('server fails an occupied port and releases its listener on close',async t=>{
  const development=await fixture(t);
  await assert.rejects(startDevServer({root:development.root,port:development.server.address().port}),{code:'EADDRINUSE'});
  const port=development.server.address().port;
  await development.close();
  await development.close();
  const replacement=await startDevServer({root:development.root,port});
  await replacement.close();
});
test('public UI and procedural hero module load',async t=>{
  const root=path.join(projectRoot,'desktop/ui');
  const development=await startDevServer({root,port:0});
  t.after(()=>development.close());
  for(const resource of ['/','/app.js','/styles.css','/spatial/original-hero.mjs']) {
    const result=await get(development.url+resource);
    assert.equal(result.status,200,resource);
    assert.ok(result.body.length>100,resource);
  }
});
test('a junction or symlink cannot expose assets outside the UI root',async t=>{
  const development=await fixture(t);
  const outside=await fs.mkdtemp(path.join(os.tmpdir(),'argus-outside-ui-'));
  t.after(()=>fs.rm(outside,{recursive:true,force:true}));
  await fs.writeFile(path.join(outside,'outside.js'),'not public');
  const link=path.join(development.root,'linked');
  await fs.symlink(outside,link,process.platform==='win32'?'junction':'dir');
  // Remove the link before fixture cleanup hooks run (test hooks run FIFO).
  try {
    assert.equal((await get(development.url+'/linked/outside.js')).status,404);
  } finally {
    await fs.unlink(link);
  }
});
