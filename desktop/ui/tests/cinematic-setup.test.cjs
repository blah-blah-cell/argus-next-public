const {test,before}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
let pose,step,stops,frames,mount,map;
before(async()=>{({journeyPose:pose,setupStep:step,SETUP_STOPS:stops,setupProgress:map}=await import('../spatial/journey-pose.mjs'));({panelFrames:frames,createSetupStage:mount}=await import('../spatial/setup-stage.mjs'));});
test('three exact holds freeze hero, camera and environment independently of progress',()=>{
  const geometry=p=>Object.fromEntries(Object.entries(pose(p)).filter(([key])=>!['progress','chapter'].includes(key)));
  for(const [start,end] of [[.08,.14],[.22,.30],[.38,.46],...[ [.22,.37],[.51,.67],[.81,1] ].map(pair=>pair.map(map))]){
    const held=geometry(start);for(let p=start;p<=end;p+=.005)assert.deepEqual(geometry(p),held);
  }
  assert.equal(stops.length,3);assert.deepEqual(stops.map(step),[0,1,2]);
  const [a,b,c]=stops.map(pose);
  assert.notEqual(a.rotationY,b.rotationY);assert.notEqual(b.rotationY,c.rotationY);
  assert.notEqual(a.worldYaw,b.worldYaw);assert.notEqual(a.cameraYaw,b.cameraYaw);
});
test('scrubbing is deterministic, finite, bounded and continuous across transition boundaries',()=>{
  for(const p of [-Infinity,NaN,-10,0,.14,.22,.37,.44,.51,.67,.74,.81,1,99,Infinity]){
    const value=pose(p);assert.ok(Object.values(value).filter(x=>typeof x==='number').every(Number.isFinite));
  }
  const forward=Array.from({length:2001},(_,i)=>pose(i/2000));
  for(let i=forward.length-1;i>=0;i--){assert.deepEqual(pose(i/2000),forward[i]);if(i)for(const key of ['x','rotationY','worldYaw','cameraYaw','cameraRoll'])assert.ok(Math.abs(forward[i][key]-forward[i-1][key])<.06);}
});
test('native panel keyframes share pose hold intervals and only animate composited properties',()=>{
  for(let i=0;i<3;i++){
    const keys=frames(i);assert.equal(keys[0].opacity,0);
    assert.ok(keys.every(key=>Object.keys(key).every(name=>['opacity','transform','offset'].includes(name))));
    const held=keys.filter(key=>key.opacity===1);assert.equal(held.length,2);assert.ok(held.every(key=>key.transform.includes('rotateY(0deg)')));
    for(let j=1;j<keys.length;j++)assert.ok(keys[j].offset>=keys[j-1].offset);
  }
});
test('one pinned stage moves the real controls, gates inactive panels, and restores linear fallback',()=>{
  const previous={document:global.document,window:global.window,IntersectionObserver:global.IntersectionObserver};
  const make=()=>({inert:false,dataset:{},attrs:{},children:[],append(item){this.children.push(item);},setAttribute(k,v){this.attrs[k]=v;},removeAttribute(k){delete this.attrs[k];},focus(){},scrollIntoView(){}});
  const panels=[make(),make(),make()],footer=make(),nav=[0,1,2].map(index=>({...make(),dataset:{setupTarget:String(index)}}));
  const setup=make(),pin=make(),workspace=make(),anchors=[make(),make(),make()],bands=[-1,0,1,2].map(index=>({...make(),dataset:{journeyBand:String(index)}}));
  setup.querySelectorAll=selector=>selector.includes('onboarding-step')?panels:nav;setup.querySelector=()=>footer;
  const subject={querySelectorAll:selector=>selector.includes('intro-')?[]:selector.includes('anchor')?anchors:bands};
  let handler,cancels=0;const activated=[];
  global.document={body:{dataset:{},classList:{toggle(){},remove(){}}},documentElement:{dataset:{}},addEventListener(){},removeEventListener(){}};
  global.window={matchMedia:()=>({matches:false})};
  global.IntersectionObserver=class{constructor(fn){handler=fn;}observe(){}disconnect(){}};
  try{
    const stage=mount({setup,pin,subject,workspace,onStep:value=>activated.push(value)});
    const driver={progress:()=>stops[0],animate:()=>({cancel(){cancels++;}})};stage.configure(true,driver);
    assert.equal(pin.children.at(-1),setup);assert.equal(panels[2].children.at(-1),footer);
    assert.deepEqual(panels.map(panel=>panel.inert),[false,true,true]);
    handler([{isIntersecting:true,target:bands[2]}]);assert.equal(stage.active,1);assert.deepEqual(panels.map(panel=>panel.inert),[true,false,true]);assert.equal(nav[1].attrs['aria-current'],'step');
    stage.configure(false,driver);assert.equal(workspace.children.at(-1),setup);assert.equal(setup.children.at(-1),footer);
    assert.ok(panels.every(panel=>!panel.inert));assert.equal(cancels,4);stage.stop();
  }finally{for(const [key,value]of Object.entries(previous)){if(value===undefined)delete global[key];else global[key]=value;}}
});
test('camera and backdrop share the scene, no wheel interception or scroll-position polling is added',()=>{
  const read=name=>fs.readFileSync(path.join(__dirname,'../spatial',name),'utf8');
  const scene=read('scene.mjs'),stage=read('setup-stage.mjs'),css=read('cinematic-setup.css');
  assert.match(scene,/world.rotation.set\(pose.worldPitch,pose.worldYaw,pose.worldRoll\)/);
  assert.match(scene,/camera.rotateZ\(pose.cameraRoll\)/);
  assert.match(stage,/IntersectionObserver/);assert.doesNotMatch(stage,/scrollY|scrollTop|getBoundingClientRect|setInterval|preventDefault/);
  assert.match(css,/\.journey-cinematic #onboarding-dialog/);
});
test('first-run boot disables stale page-scroll restoration without touching returning users',()=>{
  const vm=require('node:vm'),source=fs.readFileSync(path.join(__dirname,'../onboarding-boot.js'),'utf8');
  for(const complete of [false,true]){
    const history={scrollRestoration:'auto'};
    vm.runInNewContext(source,{history,localStorage:{getItem:()=>complete?'argus-onboarding/v1':null},document:{documentElement:{classList:{add(){}}}}});
    assert.equal(history.scrollRestoration,complete?'auto':'manual');
  }
});

test('three introduction chapters precede every setup stop with no readiness event during introduction',()=>{
  for(const [p,chapter] of [[.10,'identity'],[.25,'memory'],[.41,'evidence']]){
    assert.equal(pose(p).chapter,chapter);assert.equal(step(p),-1);
    assert.ok(stops.every(stop=>stop>p));
  }
  const html=fs.readFileSync(path.join(__dirname,'../index.html'),'utf8');
  assert.equal((html.match(/data-intro-copy=/g)||[]).length,3);
  assert.equal((html.match(/data-intro-anchor=/g)||[]).length,3);
  const css=fs.readFileSync(path.join(__dirname,'../spatial/cinematic-setup.css'),'utf8');
  assert.doesNotMatch(css,/\.journey-cinematic \.spatial-narrative[^\{]*\{display:none/);
  assert.match(css,/data-setup-stage="welcome".*visibility:hidden/);
  const workspace=fs.readFileSync(path.join(__dirname,'../spatial/workspace.mjs'),'utf8');
  assert.match(workspace,/if\(step>=0\)window.dispatchEvent\(new CustomEvent\('argus:setup-stage'/);
});

test('continue visits introduction anchors in order while explicit skip can go directly to setup',()=>{
  const previous={document:global.document,window:global.window,IntersectionObserver:global.IntersectionObserver};
  const visited=[];
  const make=name=>({dataset:{},append(){},setAttribute(){},removeAttribute(){},focus(){},scrollIntoView(){visited.push(name);}});
  const panels=[0,1,2].map(i=>make(`setup-${i}`)),intro=[0,1,2].map(i=>make(`intro-${i}`));
  const setup={...make('root'),querySelector:()=>make('footer'),querySelectorAll:s=>s.includes('onboarding-step')?panels:[]};
  const subject={querySelectorAll:s=>s==='[data-intro-anchor]'?intro:s==='[data-journey-anchor]'?panels:[]};
  global.document={body:{dataset:{},classList:{toggle(){},remove(){}}},documentElement:{dataset:{}},addEventListener(){},removeEventListener(){}};
  global.window={matchMedia:()=>({matches:false})};global.IntersectionObserver=class{observe(){}disconnect(){}};
  try{
    let progress=0;const stage=mount({setup,pin:make('pin'),subject,workspace:make('workspace')});
    stage.configure(true,{progress:()=>progress,animate:()=>({cancel(){}})});
    for(const p of [0,.10,.25,.41]){progress=p;stage.next();}
    assert.deepEqual(visited,['intro-0','intro-1','intro-2','setup-0']);
    stage.goToStep(0);assert.equal(visited.at(-1),'setup-0');stage.stop();
  }finally{for(const [key,value]of Object.entries(previous)){if(value===undefined)delete global[key];else global[key]=value;}}
});
