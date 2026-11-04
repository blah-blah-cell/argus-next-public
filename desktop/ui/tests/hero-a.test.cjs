const {test,before}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
let math;before(async()=>{math=await import('../spatial/hero-math.mjs');});
const root=path.join(__dirname,'..');
test('public hero uses independent procedural geometry',()=> {
  const source=fs.readFileSync(path.join(root,'spatial/original-hero.mjs'),'utf8');
  assert.match(source,/new T\.ExtrudeGeometry/);
  assert.doesNotMatch(source,/hero-a\.geometry\.json|env-nx\.png/);
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
test('public hero cleans up its GPU resources',()=> {
  const source=fs.readFileSync(path.join(root,'spatial/original-hero.mjs'),'utf8');
  assert.match(source,/geometry.dispose\(\)/);assert.match(source,/material.dispose\(\)/);
  assert.doesNotMatch(source,/https?:\/\/|Math.random\(|setInterval|setAnimationLoop/);
});
