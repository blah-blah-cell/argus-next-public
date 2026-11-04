const {test,before}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
let math;before(async()=>{math=await import('../spatial/hero-math.mjs');});
const root=path.join(__dirname,'..'),data=JSON.parse(fs.readFileSync(path.join(root,'spatial/assets/hero-a.geometry.json')));
test('reference extraction contains exactly the requested three hero meshes',()=> {
  assert.equal(data.schema,'argus/hero-reference-geometry/v1');
  assert.deepEqual(data.meshes.map(m=>m.name),['Alche_A','Alche_Outline','Alche_SideScreen']);
  assert.equal(data.meshes[0].position.count,1588);assert.equal(data.meshes[0].index.count,6891);
  assert.equal(data.meshes[1].position.count,7690);assert.equal(data.meshes[2].position.count,4);
});
test('every extracted stream is finite and every triangle index is in bounds',()=> {
  for(const mesh of data.meshes) {
    const positions=math.decodeHeroStream(mesh.position),indices=math.decodeHeroStream(mesh.index);
    assert.equal(positions.length,mesh.position.count*3);assert.equal(indices.length%3,0);
    assert.ok(indices.every(index=>index<mesh.position.count));
    if(mesh.uv)assert.equal(math.decodeHeroStream(mesh.uv).length,mesh.position.count*2);
    if(mesh.normal)assert.equal(math.decodeHeroStream(mesh.normal).length,positions.length);
  }
});
test('loader refuses malformed, oversized and inconsistent streams',()=> {
  for(const bad of [null,{componentType:999,count:1,components:3,data:''},{componentType:5126,count:100001,components:3,data:''},{componentType:5126,count:1,components:3,data:'AAAA'}])assert.throws(()=>math.decodeHeroStream(bad));
});
test('pointer impulse settles to a true idle state instead of spinning forever',()=> {
  let state={x:0,y:0,vx:1,vy:-1};
  state=math.stepHeroMotion(state,1/30);assert.notEqual(state.x,0);assert.equal(state.active,true);
  for(let i=0;i<180;i++)state=math.stepHeroMotion(state,1/30);
  assert.deepEqual(state,{x:0,y:0,vx:0,vy:0,active:false});
});
test('motion stays bounded at pathological deltas and stops immediately for reduced motion',()=> {
  const state=math.stepHeroMotion({x:Infinity,y:NaN,vx:1e5,vy:-1e5},1e9);
  assert.ok(Number.isFinite(state.x)&&Number.isFinite(state.y));assert.ok(Math.abs(state.x)<=.45&&Math.abs(state.y)<=.6);
  assert.deepEqual(math.stepHeroMotion(state,.033,true),{x:0,y:0,vx:0,vy:0,active:false});
});
test('hero uses local assets, real refraction input and explicitly owned GPU cleanup',()=> {
  const source=fs.readFileSync(path.join(root,'spatial/hero-a.mjs'),'utf8');
  assert.match(source,/new T.WebGLRenderTarget/);assert.match(source,/uBackdrop/);assert.match(source,/new T.CubeTextureLoader/);
  assert.match(source,/group.visible=false/);assert.match(source,/backdrop.dispose/);assert.match(source,/glass.dispose/);
  assert.doesNotMatch(source,/plateScene|plateCamera|fillText/);
  assert.doesNotMatch(source,/https?:\/\/|Math.random\(|setInterval|setAnimationLoop/);
  const notice=fs.readFileSync(path.join(root,'spatial/assets/NOTICE.md'),'utf8');
  assert.match(notice,/No redistribution licence/);
});
