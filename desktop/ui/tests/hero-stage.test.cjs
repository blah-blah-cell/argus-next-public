const {test,before}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
let math,createHeroStage;
before(async()=>{math=await import('../spatial/stage-math.mjs');({createHeroStage}=await import('../spatial/hero-stage.mjs'));});
test('wall is genuinely concave, symmetric and entirely behind the hero',()=> {
  const middle=math.curveWallPoint(.5,.5),left=math.curveWallPoint(0,.5),right=math.curveWallPoint(1,.5);
  assert.equal(middle[2],-6.2);assert.equal(left[2],right[2]);assert.equal(left[0],-right[0]);assert.ok(left[2]>middle[2]);
  for(let u=0;u<=100;u++)for(const v of [0,.5,1]) {
    const p=math.curveWallPoint(u/100,v);assert.ok(p.every(Number.isFinite));assert.ok(p[2]<-1);assert.ok(p[1]>=-2.15);
  }
  for(const [u,v] of [[NaN,Infinity],[-1e9,1e9]])assert.ok(math.curveWallPoint(u,v).every(Number.isFinite));
});
test('pointer damping converges exactly and stops scheduling rather than drifting forever',()=> {
  let state={x:0,y:0,alpha:0,focus:0};const target={x:.7,y:-.6,alpha:1,focus:1};
  state=math.stepStageMotion(state,target,1/30);assert.ok(state.x>0&&state.x<.7);assert.equal(state.active,true);
  for(let i=0;i<100;i++)state=math.stepStageMotion(state,target,1/30);
  assert.deepEqual(state,{...target,active:false});
  for(let i=0;i<100;i++)state=math.stepStageMotion(state,{},1/30);
  assert.deepEqual(state,{x:0,y:0,alpha:0,focus:0,active:false});
});
test('pointer state is finite and bounded for malformed events, time jumps and reduced motion',()=> {
  const state=math.stepStageMotion({x:Infinity,y:NaN,alpha:100,focus:-9},{x:1e9,y:-1e9,alpha:9,focus:9},1e9);
  for(const key of ['x','y','alpha','focus'])assert.ok(Number.isFinite(state[key]));
  assert.ok(state.x<=1&&state.y>=-1&&state.alpha<=1&&state.focus<=1);
  assert.deepEqual(math.stepStageMotion(state,{x:1,alpha:1},1,true),{x:0,y:0,alpha:0,focus:0,active:false});
});
function stageFixture() {
  const stats={textures:0,geometries:0,materials:0,draws:0};
  const host={children:[],append(child){this.children.push(child);}};
  const flags={fine:true,reduced:false},root={dataset:{motion:'full'}};
  const cursor={style:{},setAttribute(){},remove(){host.children=host.children.filter(x=>x!==this);}};
  const canvas={width:0,height:0,getContext(){return {fillRect(){},fillText(){stats.draws++;},measureText(){return {width:280};}};}};
  class Group {constructor(){this.children=[];this.rotation={};}add(item){this.children.push(item);}}
  class PlaneGeometry {
    constructor(){const values=[[-.5,-.5,0],[0,0,0],[.5,.5,0]];this.attributes={position:{count:3,getX:i=>values[i][0],getY:i=>values[i][1],setXYZ(i,...p){values[i]=p;}}};}
    computeVertexNormals(){}dispose(){stats.geometries++;}
  }
  class ShaderMaterial {constructor(options){Object.assign(this,options);}dispose(){stats.materials++;}}
  class Mesh {constructor(geometry,material){Object.assign(this,{geometry,material,rotation:{},position:{}});}}
  class Vector3 {constructor(x,y,z){this.set(x,y,z);}set(x,y,z){Object.assign(this,{x,y,z});}}
  class Vector2 {constructor(x,y){this.set(x,y);}set(x,y){Object.assign(this,{x,y});}}
  class CanvasTexture {constructor(image){this.image=image;}dispose(){stats.textures++;}}
  const T={Group,PlaneGeometry,ShaderMaterial,Mesh,Vector2,Vector3,CanvasTexture,NoColorSpace:'none',DoubleSide:2};
  const prior={document:global.document,window:global.window,getComputedStyle:global.getComputedStyle};
  let studio;const outCanvas={dataset:{}};
  global.document={documentElement:root,querySelector(){return {textContent:'ARGUS'};},createElement(name){return name==='canvas'?canvas:cursor;}};
  global.window={matchMedia(query){return {get matches(){return query.includes('reduced')?flags.reduced:flags.fine;}};}};
  global.getComputedStyle=()=>({fontFamily:'Arial'});
  try {studio=createHeroStage(T,host,outCanvas);}catch(error){Object.assign(global,prior);throw error;}
  return {studio,stats,host,flags,root,cursor,canvas,outCanvas,cleanup(){Object.assign(global,prior);}};
}
test('stage owns one power-of-two wordmark texture and releases all of its resources',()=> {
  const f=stageFixture();try {
    assert.deepEqual([f.canvas.width,f.canvas.height],[2048,1024]);assert.equal(f.stats.draws,5);
    assert.equal(f.studio.group.children.length,2);assert.equal(f.host.children.length,1);
    f.studio.dispose();assert.deepEqual(f.stats,{textures:1,geometries:2,materials:2,draws:5});assert.equal(f.host.children.length,0);
  }finally{f.cleanup();}
});
test('pointer lighting reacts to fine input, then disappears in disabled and reduced-motion states',()=> {
  const f=stageFixture();try {
    f.studio.resize(1000,700);f.studio.setEnabled(true);f.studio.pointer(.8,-.5,true);f.studio.tick(100,0);
    assert.ok(f.studio.pointerLight.x>0);assert.equal(f.cursor.hidden,false);assert.match(f.cursor.style.transform,/translate3d/);
    f.root.dataset.motion='reduced';f.studio.tick(130,0);assert.equal(f.cursor.hidden,true);assert.equal(f.studio.pointerLight.z,0);
    f.root.dataset.motion='full';f.studio.setEnabled(false);f.studio.tick(160,0);assert.equal(f.cursor.hidden,true);assert.equal(f.studio.pointerLight.z,0);
    f.studio.dispose();
  }finally{f.cleanup();}
});
test('touch/coarse input retains a static studio without an artificial cursor',()=> {
  const f=stageFixture();try {
    f.flags.fine=false;f.studio.setEnabled(true);f.studio.pointer(1,1,true);assert.equal(f.studio.tick(100,0),false);
    assert.equal(f.cursor.hidden,true);assert.equal(f.studio.pointerLight.z,0);assert.equal(f.outCanvas.dataset.backdrop,'curved-wall');f.studio.dispose();
  }finally{f.cleanup();}
});

test('surrounding lettering persists during travel and pauses when disabled',()=> {
  const f=stageFixture();try {
    f.studio.resize(1000,700);f.studio.setEnabled(true);
    const uniforms=f.studio.group.children[0].material.uniforms;
    assert.equal(f.studio.tick(100,0),true);
    const first=uniforms.uTime.value;
    f.studio.tick(140,0);assert.ok(uniforms.uTime.value>first);assert.equal(uniforms.uMotion.value,1);
    assert.equal(f.outCanvas.dataset.textMotion,'active');
    assert.equal(f.studio.tick(180,.6),true);assert.ok(uniforms.uWordOpacity.value>.07);
    f.studio.setEnabled(false);const paused=uniforms.uTime.value;f.studio.tick(40000,.6);assert.equal(uniforms.uTime.value,paused);
    f.studio.dispose();
  }finally{f.cleanup();}
});

test('reduced, disabled and coarse stages do not advance text phase; resuming is bounded',()=> {
  const f=stageFixture();try {
    f.studio.setEnabled(true);f.studio.tick(100,0);
    const uniforms=f.studio.group.children[0].material.uniforms,phase=uniforms.uTime.value;
    f.root.dataset.motion='reduced';assert.equal(f.studio.tick(140,0),false);
    assert.equal(uniforms.uMotion.value,0);assert.equal(uniforms.uTime.value,phase);
    f.root.dataset.motion='full';f.flags.fine=false;assert.equal(f.studio.tick(180,0),false);
    assert.equal(uniforms.uTime.value,phase);
    f.flags.fine=true;f.studio.setEnabled(false);assert.equal(f.outCanvas.dataset.textMotion,'static');assert.equal(f.studio.tick(220,0),false);
    f.studio.setEnabled(true);f.studio.tick(999999,0);
    assert.ok(uniforms.uTime.value-phase<=.050001);f.studio.dispose();
  }finally{f.cleanup();}
});

test('letter-only cursor warp uses aspect-correct screen distance and eases out over controls',()=> {
  const f=stageFixture();try {
    f.outCanvas.width=2000;f.outCanvas.height=1400;
    f.studio.resize(1000,700);f.studio.setEnabled(true);f.studio.pointer(.6,-.2,true);
    for(let i=0;i<100;i++)f.studio.tick(100+i*33,0);
    const wall=f.studio.group.children[0].material;
    assert.equal(wall.uniforms.uViewport.value.x,2000);assert.equal(wall.uniforms.uViewport.value.y,1400);
    assert.equal(wall.uniforms.uFocus.value,1);
    assert.match(wall.fragmentShader,/gl_FragCoord.xy\/uViewport/);
    assert.match(wall.fragmentShader,/uViewport.x\/uViewport.y/);
    assert.match(wall.fragmentShader,/uMotion\*uPointer.z\*\(1.-uFocus\)\*influence/);
    assert.match(wall.fragmentShader,/typeUv\+=cursorDelta/);f.studio.dispose();
  }finally{f.cleanup();}
});

test('magnetic lettering has no traveling ripple or per-letter wobble',()=> {
  const f=stageFixture();try {
    const shader=f.studio.group.children[0].material.fragmentShader;
    assert.match(shader,/typeUv\+=cursorDelta\*vec2\(\.12,\.12\)\*warp/);
    assert.match(shader,/vec2\(sin\(uTime\*\.18\),cos\(uTime\*\.14\)\)\*\.0015\*uMotion/);
    assert.doesNotMatch(shader,/sin\(radius|sin\(typeUv|cos\(typeUv/);
    // Analytic maximum of r*exp(-75*r*r)*.12, before existing motion/focus gates.
    const radius=1/Math.sqrt(150),maxDisplacement=radius*Math.exp(-75*radius*radius)*.12;
    assert.ok(maxDisplacement<.006);
    f.studio.dispose();
  }finally{f.cleanup();}
});
test('the real curved lettering feeds refraction and replaces the wireframe/flat-copy render paths',()=> {
  const read=name=>fs.readFileSync(path.join(__dirname,'../spatial',name),'utf8');
  const scene=read('scene.mjs'),hero=read('hero-a.mjs'),stage=read('hero-stage.mjs'),css=read('workspace.css');
  assert.match(scene,/world.add\(studio.group\)/);assert.match(scene,/studio.tick\(time,Math.min\(1,pose.progress\*2\),yaw\)/);assert.match(hero,/renderer.render\(scene,camera\)/);
  assert.match(scene,/world.rotation.set\(pose.worldPitch,pose.worldYaw,pose.worldRoll\)/);
  assert.doesNotMatch(scene,/GridHelper|BoxGeometry|EdgesGeometry|IcosahedronGeometry|QuadraticBezierCurve3/);
  assert.doesNotMatch(hero,/plateScene|plateCamera|fillText/);assert.match(hero,/uPointerLight/);
  assert.match(css,/\.spatial-pointer[^}]*pointer-events:none/);assert.match(css,/data-motion="reduced"/);
  assert.doesNotMatch(stage,/requestAnimationFrame|setInterval|setTimeout|Math.random|getBoundingClientRect/);
});
