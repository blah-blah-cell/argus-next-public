const {test,before}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
let createHeroVideo;before(async()=>{({createHeroVideo}=await import('../spatial/hero-video.mjs'));});
const settle=()=>new Promise(resolve=>setImmediate(resolve));
function fixture({frames=true,reject=false,deferred=false}={}) {
  const events=new Map(),requests=new Map(),states=[];
  let index=0,pauses=0,plays=0,draws=0,wakes=0,loads=0,removed=false,resolvePlay;
  const video={paused:true,readyState:2,src:'local.mp4',
    addEventListener(name,fn){events.set(name,fn);},removeEventListener(name){events.delete(name);},setAttribute(){},
    removeAttribute(name){delete this[name];},remove(){removed=true;},load(){loads++;},
    pause(){this.paused=true;pauses++;},
    play(){plays++;this.paused=false;return deferred?new Promise(resolve=>{resolvePlay=resolve;}):reject?Promise.reject(Error('Autoplay blocked')):Promise.resolve();},
    ...(frames?{requestVideoFrameCallback(fn){requests.set(++index,fn);return index;},cancelVideoFrameCallback(id){requests.delete(id);}}:{}),
  };
  const controller=createHeroVideo(video,{onStatus(state){states.push(state);},onFrame(){draws++;},invalidate(){wakes++;}});
  return {video,controller,events,requests,states,
    frame(){const entry=requests.entries().next().value;assert.ok(entry);requests.delete(entry[0]);entry[1]();},
    resolve(){resolvePlay();},stats:()=>({pauses,plays,draws,wakes,loads,removed})};
}
test('decorative video starts as a still and is muted, inline and looping',()=> {
  const f=fixture();assert.equal(f.stats().plays,0);assert.deepEqual(f.states,['poster']);
  assert.equal(f.video.muted,true);assert.equal(f.video.defaultMuted,true);assert.equal(f.video.playsInline,true);assert.equal(f.video.loop,true);
  assert.equal(f.video.preload,'none');assert.equal(f.controller.needsPaintLoop,false);f.controller.dispose();
});
test('one decoded frame wakes one demand draw; no duplicate playback or callback chain',async()=> {
  const f=fixture();f.controller.setEnabled(true);await settle();
  assert.equal(f.stats().plays,1);assert.equal(f.requests.size,1);assert.equal(f.controller.needsPaintLoop,false);
  f.controller.setEnabled(true);assert.equal(f.stats().plays,1);
  f.frame();assert.equal(f.stats().draws,1);assert.equal(f.requests.size,1);assert.equal(f.states.at(-1),'playing');
  f.controller.setEnabled(false);assert.equal(f.video.paused,true);assert.equal(f.requests.size,0);assert.equal(f.states.at(-1),'paused');
  f.controller.setEnabled(true);await settle();assert.equal(f.stats().plays,2);assert.equal(f.requests.size,1);f.controller.dispose();
});
test('an in-flight play cannot restart video after hide/disposal',async()=> {
  for(const action of ['hide','dispose']) {
    const f=fixture({deferred:true});f.controller.setEnabled(true);
    if(action==='hide')f.controller.setEnabled(false);else f.controller.dispose();
    f.resolve();await settle();assert.equal(f.video.paused,true);assert.equal(f.requests.size,0);
    if(action==='hide')f.controller.dispose();
  }
});
test('autoplay rejection is contained and retries only upon explicit interaction',async()=> {
  const f=fixture({reject:true});f.controller.setEnabled(true);await settle();
  assert.equal(f.states.at(-1),'blocked');assert.equal(f.video.paused,true);assert.equal(f.requests.size,0);
  f.controller.setEnabled(true);assert.equal(f.stats().plays,1);
  f.controller.retry();await settle();assert.equal(f.stats().plays,2);assert.equal(f.states.at(-1),'blocked');f.controller.dispose();
});
test('unsupported or failed media stays static without killing the scene',async()=> {
  const f=fixture();f.controller.setEnabled(true);await settle();f.events.get('error')();
  assert.equal(f.video.paused,true);assert.equal(f.requests.size,0);assert.equal(f.states.at(-1),'error');
  f.controller.setEnabled(true);f.controller.retry();await settle();assert.equal(f.stats().plays,1);f.controller.dispose();
});
test('older browser fallback uses the scene capped loop only while playing',async()=> {
  const f=fixture({frames:false});f.controller.setEnabled(true);await settle();assert.equal(f.controller.needsPaintLoop,true);
  f.controller.setEnabled(false);assert.equal(f.controller.needsPaintLoop,false);f.controller.dispose();
});
test('dispose cancels frames, removes listeners and releases the media resource',async()=> {
  const f=fixture();f.controller.setEnabled(true);await settle();f.controller.dispose();
  assert.equal(f.events.size,0);assert.equal(f.requests.size,0);assert.equal(f.video.src,undefined);
  assert.equal(f.stats().loads,1);assert.equal(f.stats().removed,true);
  f.controller.setEnabled(true);assert.equal(f.stats().plays,1);
});
test('local video/poster bytes match provenance and generated imagery is not telemetry',()=> {
  const assets=path.join(__dirname,'../spatial/assets'),record=JSON.parse(fs.readFileSync(path.join(assets,'PACKET_VIDEO.json')));
  for(const [name,key] of [['packet-reflection.mp4','video_sha256'],['packet-reflection-poster.jpg','poster_sha256']])
    assert.equal(crypto.createHash('sha256').update(fs.readFileSync(path.join(assets,name))).digest('hex'),record[key]);
  assert.match(record.purpose,/Never observed telemetry/);assert.equal(record.duration_seconds,10);
});
test('renderer binds directional video optics and pauses on reduced motion and all scene gates',()=> {
  const read=name=>fs.readFileSync(path.join(__dirname,'../spatial',name),'utf8');
  const hero=read('hero-a.mjs'),scene=read('scene.mjs');
  assert.match(hero,/new T.VideoTexture\(video\)/);assert.match(hero,/packetLight\(reflectedDirection\)/);assert.match(hero,/packetLight\(refract\(/);
  assert.match(hero,/if\(reduced\)\{uniforms.uPacketVideo.value=poster/);assert.match(hero,/playback.setEnabled\(enabled&&!reduced\)/);
  assert.match(hero,/sRGBTransferEOTF\(vec4\(color,1\.\)\)/);
  assert.match(hero,/attributeFilter:\['data-motion'\]/);assert.match(hero,/videoTexture.dispose\(\)/);
  assert.match(scene,/enabled&&visible&&!lost&&!destroyed&&!document.hidden/);
  assert.doesNotMatch(read('hero-video.mjs'),/setInterval|setTimeout|requestAnimationFrame|Math.random/);
});
