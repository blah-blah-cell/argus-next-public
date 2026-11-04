import { clampCamera, pixelRatio, storyPose } from './core.mjs?v=room-orbit-1';
import {createHeroA} from './original-hero.mjs';
import {createHeroStage} from './hero-stage.mjs?v=room-orbit-1';

export async function createScene(canvas, onStatus) {
  const T = await import('../vendor/three/three.module.js');
  let renderer;
  try {
    renderer = new T.WebGLRenderer({ canvas, alpha: true, antialias: true,
      powerPreference: 'low-power', failIfMajorPerformanceCaveat: true });
  } catch (error) {
    throw new Error(`3D acceleration unavailable: ${error.message}`);
  }
  renderer.outputColorSpace = T.SRGBColorSpace;
  renderer.toneMapping = T.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;
  const scene = new T.Scene();
  const camera = new T.PerspectiveCamera(40, 1, .1, 90);
  const rig = { yaw: .32, pitch: .12, distance: 10.4 };
  const world = new T.Group();
  scene.add(world);
  const host=canvas.closest('.spatial-stage');
  const studio=createHeroStage(T,host,canvas);world.add(studio.group);
  let hero;
  try {hero=await createHeroA(T,renderer,{invalidate,onVideoStatus(status){canvas.dataset.reflection=status;}});}
  catch(error){studio.dispose();renderer.dispose();throw error;}
  const mark=hero.group;world.add(mark);
  // Identity scene only. Observed topology belongs in the data workspace,
  // not between the camera and the hero or inside its refraction backdrop.
  let width=1, height=1, bounds=null, frameId=0, lastPaint=-100;
  let visible=true, enabled=true, destroyed=false, lost=false, painted=false, paintCount=0;
  let drag=null;
  let readStory=()=>0;
  function syncHero() {hero.setEnabled(enabled&&visible&&!lost&&!destroyed&&!document.hidden);studio.setEnabled(enabled&&visible&&!lost&&!destroyed&&!document.hidden);}

  function paint(time=0) {
    frameId=0;
    if (destroyed || lost || !enabled || !visible || document.hidden) return;
    if (time-lastPaint < 32) { frameId=requestAnimationFrame(paint); return; }
    lastPaint=time;
    const pose=storyPose(readStory());
    world.rotation.set(pose.worldPitch,pose.worldYaw,pose.worldRoll);
    canvas.dataset.storyProgress=pose.progress.toFixed(3);
    const c=clampCamera(rig);
    const yaw=c.yaw+pose.cameraYaw,pitch=c.pitch+pose.cameraPitch;
    const distance=c.distance+pose.cameraDolly;
    // Keep the subject framed left as the camera travels around the room.
    mark.position.set(pose.x*Math.cos(yaw),pose.y,-pose.x*Math.sin(yaw));
    mark.rotation.set(0,pose.rotationY+pose.cameraYaw,pose.rotationZ);mark.scale.setScalar(pose.scale);
    camera.position.set(Math.sin(yaw)*Math.cos(pitch)*distance,
      Math.sin(pitch)*distance+.25, Math.cos(yaw)*Math.cos(pitch)*distance);
    camera.lookAt(0,.1,0);camera.rotateZ(pose.cameraRoll);camera.updateMatrixWorld();
    canvas.dataset.roomYaw=pose.cameraYaw.toFixed(3);
    const studioMoving=studio.tick(time,Math.min(1,pose.progress*2),yaw);hero.setLight(studio.pointerLight);
    const moving=hero.tick(time)||studioMoving;
    try { hero.prepare(scene,camera);renderer.render(scene,camera); }
    catch (error) { lost=true;syncHero();onStatus('failed');return; }
    canvas.dataset.frames=String(++paintCount);
    canvas.dataset.hero='procedural';
    if (!painted) { painted=true; onStatus('webgl'); }
    if(moving)invalidate();
  }
  function invalidate() {
    if (!frameId && !destroyed && !lost && visible && enabled && !document.hidden) frameId=requestAnimationFrame(paint);
  }
  function resize() {
    const rect=canvas.parentElement.getBoundingClientRect(); bounds=canvas.getBoundingClientRect();
    width=Math.max(1,rect.width); height=Math.max(1,rect.height);
    renderer.setPixelRatio(pixelRatio(width,height,window.devicePixelRatio));
    renderer.setSize(width,height,false); camera.aspect=width/height; camera.updateProjectionMatrix();
    hero.resize(width,height);
    studio.resize(width,height);
    invalidate();
  }
  function view(preset) {
    hero.retryVideo();
    Object.assign(rig, preset==='top'?{yaw:0,pitch:1.05,distance:12}:preset==='front'?{yaw:0,pitch:0,distance:11}:{yaw:.32,pitch:.12,distance:10.4});
    invalidate();
  }
  canvas.addEventListener('pointerdown', event=> {
    if (!enabled || lost || event.button!==0) return;
    hero.retryVideo();
    bounds=canvas.getBoundingClientRect(); drag={x:event.clientX,y:event.clientY};
    canvas.setPointerCapture(event.pointerId); canvas.classList.add('is-dragging');
  });
  canvas.addEventListener('pointermove', event=> {
    if(!enabled)return;
    if(!drag){if(bounds){hero.hover((event.clientX-bounds.left)/width*2-1,(event.clientY-bounds.top)/height*2-1);invalidate();}return;}
    const dx=event.clientX-drag.x, dy=event.clientY-drag.y;
    rig.yaw-=dx*.006; rig.pitch=Math.max(-1.2,Math.min(1.2,rig.pitch+dy*.005));
    drag={x:event.clientX,y:event.clientY}; invalidate();
  });
  function release(event) {
    if (!drag) return;
    drag=null; canvas.classList.remove('is-dragging');
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  }
  canvas.addEventListener('pointerup',release);
  canvas.addEventListener('pointerenter',()=>{bounds=canvas.getBoundingClientRect();hero.leave();});
  canvas.addEventListener('pointerleave',()=>{hero.leave();invalidate();});
  canvas.addEventListener('pointercancel',()=>{drag=null;canvas.classList.remove('is-dragging');});
  function pointerLight(event) {
    if(!enabled||lost||event.pointerType==='touch')return;
    if(!bounds)bounds=canvas.getBoundingClientRect();
    studio.pointer((event.clientX-bounds.left)/width*2-1,(event.clientY-bounds.top)/height*2-1,!!event.target.closest('button,select,a,input,summary'));
    invalidate();
  }
  function pointerEnter(){bounds=canvas.getBoundingClientRect();}
  function pointerLeave(){studio.leave();hero.leave();invalidate();}
  function scrollBounds(){bounds=null;}
  host.addEventListener('pointermove',pointerLight);
  host.addEventListener('pointerenter',pointerEnter);
  host.addEventListener('pointerleave',pointerLeave);
  document.addEventListener('scroll',scrollBounds,{passive:true});
  canvas.addEventListener('wheel',event=>{
    if (!event.ctrlKey || !enabled) return;
    event.preventDefault(); rig.distance=clampCamera({ ...rig,distance:rig.distance+event.deltaY*.008 }).distance; invalidate();
  },{passive:false});
  canvas.addEventListener('webglcontextlost',event=>{
    event.preventDefault(); lost=true; if(frameId) cancelAnimationFrame(frameId);frameId=0;
    syncHero();
    onStatus('lost');
  });
  const resizeObserver=new ResizeObserver(resize); resizeObserver.observe(canvas.parentElement);
  const intersection=new IntersectionObserver(entries=>{
    visible=entries[0].isIntersecting;
    syncHero();
    if (!visible && frameId) {cancelAnimationFrame(frameId);frameId=0;} else invalidate();
  }); intersection.observe(canvas);
  function visibility() {syncHero();if(document.hidden && frameId){cancelAnimationFrame(frameId);frameId=0;} else invalidate();}
  document.addEventListener('visibilitychange',visibility);
  resize();
  return {
    view, invalidate,
    setRoughness(value){hero.setRoughness(value);},
    setStoryReader(reader) {readStory=reader;invalidate();},
    zoom(delta) {rig.distance=clampCamera({...rig,distance:rig.distance+delta}).distance;invalidate();},
    rotate(delta) {rig.yaw+=delta;invalidate();},
    setEnabled(value) {enabled=value;syncHero();if(!value && frameId){cancelAnimationFrame(frameId);frameId=0;}else invalidate();},
    destroy() {
      destroyed=true;if(frameId)cancelAnimationFrame(frameId);
      resizeObserver.disconnect();intersection.disconnect();document.removeEventListener('visibilitychange',visibility);
      host.removeEventListener('pointermove',pointerLight);host.removeEventListener('pointerenter',pointerEnter);host.removeEventListener('pointerleave',pointerLeave);document.removeEventListener('scroll',scrollBounds);
      hero.dispose();studio.dispose();renderer.dispose();
    },
  };
}
