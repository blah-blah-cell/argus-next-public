const {test,before}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
let camera,frames,stations,wall;
before(async()=>{({roomCamera:camera,roomUiFrames:frames,ROOM_STATIONS:stations}=await import('../spatial/room-motion.mjs'));({roomWallPoint:wall}=await import('../spatial/stage-math.mjs'));});
test('room camera travels around the subject, with six exactly still readable stations',()=>{
  assert.ok(camera(1).cameraYaw>Math.PI);
  for(const [from,to] of [[.08,.14],[.22,.30],[.38,.46],[.5944,.6724],[.7452,.8284],[.9012,1]]){
    const held=camera(from);for(let p=from;p<=to;p+=.001)assert.deepEqual(camera(p),held);
    assert.equal(held.cameraDolly,0);assert.equal(held.cameraRoll,0);
  }
  let previous=camera(0);
  for(let i=1;i<=2000;i++){
    const p=i/2000,next=camera(p);assert.deepEqual(next,camera(p));
    assert.ok(next.cameraYaw>=previous.cameraYaw);assert.ok(Math.abs(next.cameraYaw-previous.cameraYaw)<.02);
    assert.ok(next.cameraDolly>=0&&next.cameraDolly<=1.65);assert.ok(Math.abs(next.cameraRoll)<=.075);previous=next;
  }
  for(const p of [NaN,Infinity,-Infinity,-99,99])assert.ok(Object.values(camera(p)).every(Number.isFinite));
});
test('overlay uses the same exact camera schedule and only compositor transforms',()=>{
  const keys=frames();assert.ok(keys.length>=201);
  for(const [p]of stations){const key=keys.find(k=>k.offset===p);assert.ok(key);assert.ok(key.transform.includes('perspective(1600px)'));}
  assert.ok(keys.every(key=>Object.keys(key).every(k=>['offset','transform'].includes(k))));
  for(const p of [.10,.25,.41,.6256,.7868,.948]){
    const key=keys.find(k=>Math.abs(k.offset-p)<.005);assert.ok(key);assert.match(key.transform,/translate3d\(0,0px,0px\) rotateZ\(0deg\) rotateX\(0deg\)/);
  }
});
test('panoramic wall surrounds every camera angle without a front wireframe or open seam',()=>{
  for(let i=0;i<=360;i++)for(const v of [0,.5,1]){
    const [x,y,z]=wall(i/360,v);assert.ok(Math.abs(Math.hypot(x,z)-18)<1e-10);assert.ok(y>=-2.15);
  }
  const a=wall(0,.5),b=wall(1,.5);assert.ok(a.every((x,i)=>Math.abs(x-b[i])<1e-10));
  const read=name=>fs.readFileSync(path.join(__dirname,'../spatial',name),'utf8');
  assert.doesNotMatch(read('hero-stage.mjs'),/group.rotation.y=cameraYaw/);
  assert.match(read('hero-stage.mjs'),/uWordOpacity.value=1-t\*\.92/);
  assert.match(read('setup-stage.mjs'),/driver.animate\(setup,roomUiFrames\(\)\)/);
  assert.match(read('setup-stage.mjs'),/driver.animate\(narrative,roomUiFrames\(\)\)/);
  assert.doesNotMatch(read('room-motion.mjs'),/scrollY|scrollTop|getBoundingClientRect|requestAnimationFrame|setInterval/);
});
