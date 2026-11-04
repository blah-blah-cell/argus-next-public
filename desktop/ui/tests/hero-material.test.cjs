const {test,before}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
let bind,math,KEY;before(async()=>{math=await import('../spatial/hero-math.mjs');({bindHeroRoughness:bind,ROUGHNESS_KEY:KEY}=await import('../spatial/hero-material-controls.mjs'));});
function fixture(saved=null,broken=false){
  const handlers=new Map(),writes=[],applied=[];
  const input={value:'18',disabled:true,attrs:{},setAttribute(k,v){this.attrs[k]=v;},addEventListener(k,v){handlers.set(k,v);},removeEventListener(k){handlers.delete(k);}};
  const output={textContent:''};
  const storage={getItem(){if(broken)throw Error('blocked');return saved;},setItem(k,v){if(broken)throw Error('blocked');writes.push([k,v]);}};
  const controls=bind(input,output,value=>applied.push(value),storage);
  return {input,output,writes,applied,controls,handlers,change(value){input.value=value;handlers.get('input')();}};
}
test('roughness defaults and malformed values are finite and bounded',()=>{
  assert.equal(math.clampHeroRoughness(-1),0);assert.equal(math.clampHeroRoughness(2),1);
  for(const value of [NaN,Infinity,undefined,'bad'])assert.equal(math.clampHeroRoughness(value),.18);
  for(const saved of [null,'','bad'])assert.equal(fixture(saved).input.value,'18');
});
test('saved roughness applies immediately; input updates shader callback and persistence',()=>{
  const f=fixture('65');assert.equal(f.applied[0],.65);assert.equal(f.output.textContent,'65%');
  f.change('100');assert.equal(f.applied.at(-1),1);assert.deepEqual(f.writes.at(-1),[KEY,'100']);assert.match(f.input.attrs['aria-valuetext'],/frosted/);
  f.change('0');assert.equal(f.applied.at(-1),0);assert.match(f.input.attrs['aria-valuetext'],/polished/);
});
test('blocked storage does not prevent session controls; Flat disables and disposal removes listener',()=>{
  const f=fixture(null,true);f.controls.setEnabled(true);assert.equal(f.input.disabled,false);
  f.change('42');assert.equal(f.applied.at(-1),.42);
  f.controls.setEnabled(false);assert.equal(f.input.disabled,true);assert.match(f.input.title,/3D view/);
  f.controls.dispose();assert.equal(f.handlers.size,0);
});
test('slider has native bounds and real glass shader roughness paths',()=>{
  const read=name=>fs.readFileSync(path.join(__dirname,'..',name),'utf8');
  const html=read('index.html'),hero=read('spatial/hero-a.mjs'),workspace=read('spatial/workspace.mjs');
  assert.match(html,/for="hero-roughness"/);assert.match(html,/id="hero-roughness" type="range" min="0" max="100" step="1"/);
  assert.match(hero,/uRoughness\*uRoughness\*\.04/);assert.match(hero,/environmentLight\(reflectedDirection\)/);
  assert.match(hero,/mix\(240\.,18\.,uRoughness\)/);assert.match(workspace,/scene.setRoughness\(value\)/);
  assert.match(workspace,/materialControls\?\.setEnabled\(!flat\)/);assert.match(workspace,/materialControls\?\.dispose\(\)/);
});
