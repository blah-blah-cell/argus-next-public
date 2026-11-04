import {decodeHeroStream,stepHeroMotion,clampHeroRoughness,DEFAULT_HERO_ROUGHNESS} from './hero-math.mjs?v=roughness-1';
import {createHeroVideo} from './hero-video.mjs';

// New, local renderer for the selected reference mesh. No reference app code.
const glassVertex=`
varying vec3 vWorld, vNormalView;
varying vec2 vUv;
void main(){
  vWorld=(modelMatrix*vec4(position,1.)).xyz;
  vNormalView=normalize(normalMatrix*normal);
  vUv=uv;
  gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);
}`;
const glassFragment=`
uniform sampler2D uBackdrop, uNoise, uPacketVideo;
uniform samplerCube uEnvironment;
uniform float uDecodeVideo,uRoughness;
uniform vec2 uResolution;
uniform mat3 uViewToWorld;
uniform vec3 uPointerLight;
varying vec3 vWorld, vNormalView;
varying vec2 vUv;
// A wide virtual lightbox is sampled by optical direction, never by screen UV.
// This ordinary landscape clip is not falsely treated as a 360-degree panorama.
vec3 packetLight(vec3 direction){
  vec2 uv=.5+direction.xy/max(.45,abs(direction.z))*vec2(.48,.9);
  float mask=smoothstep(0.,.12,uv.x)*(1.-smoothstep(.88,1.,uv.x));
  mask*=smoothstep(0.,.1,uv.y)*(1.-smoothstep(.9,1.,uv.y));
  float spread=uRoughness*uRoughness*.075;
  vec3 color=texture2D(uPacketVideo,clamp(uv,.001,.999)).rgb;
  color+=texture2D(uPacketVideo,clamp(uv+vec2(spread,0.),.001,.999)).rgb;
  color+=texture2D(uPacketVideo,clamp(uv-vec2(spread,0.),.001,.999)).rgb;
  color+=texture2D(uPacketVideo,clamp(uv+vec2(0.,spread),.001,.999)).rgb;
  color+=texture2D(uPacketVideo,clamp(uv-vec2(0.,spread),.001,.999)).rgb;
  color/=5.;
  // Video samplers need explicit sRGB decoding in a custom ShaderMaterial.
  color=mix(color,sRGBTransferEOTF(vec4(color,1.)).rgb,uDecodeVideo);
  return color*mask;
}
vec3 environmentLight(vec3 direction){
  vec3 axis=abs(direction.y)<.9?vec3(0.,1.,0.):vec3(1.,0.,0.);
  vec3 tangent=normalize(cross(direction,axis))*uRoughness*uRoughness*.3;
  vec3 bitangent=cross(direction,tangent);
  return (textureCube(uEnvironment,direction).rgb+
    textureCube(uEnvironment,normalize(direction+tangent)).rgb+
    textureCube(uEnvironment,normalize(direction-tangent)).rgb+
    textureCube(uEnvironment,normalize(direction+bitangent)).rgb+
    textureCube(uEnvironment,normalize(direction-bitangent)).rgb)/5.;
}
void main(){
  vec3 normal=normalize(vNormalView);
  vec3 normalWorld=normalize(uViewToWorld*normal);
  vec3 incident=normalize(vWorld-cameraPosition);
  vec2 screen=gl_FragCoord.xy/uResolution;
  float grain=texture2D(uNoise,vUv*9.).r;
  vec2 bend=normal.xy*(1.-abs(normal.z)*.7)*.1;
  vec3 transmitted=vec3(0.);
  for(int i=0;i<8;i++){
    float angle=float(i)*2.399963+grain*6.283185*uRoughness;
    vec2 spread=vec2(cos(angle),sin(angle))*sqrt((float(i)+.5)/8.)*(.001+grain*.0015+uRoughness*uRoughness*.04);
    vec2 base=screen+spread;
    transmitted.r+=texture2D(uBackdrop,clamp(base-bend*1.04,.001,.999)).r;
    transmitted.g+=texture2D(uBackdrop,clamp(base-bend*1.15,.001,.999)).g;
    transmitted.b+=texture2D(uBackdrop,clamp(base-bend*1.42,.001,.999)).b;
  }
  transmitted/=8.;
  float facing=abs(dot(-incident,normalWorld));
  float fresnel=.07+.93*pow(1.-facing,5.);
  vec3 reflectedDirection=reflect(incident,normalWorld);
  vec3 reflection=environmentLight(reflectedDirection);
  vec3 packets=packetLight(reflectedDirection);
  vec3 throughPackets=packetLight(refract(incident,normalWorld,1./1.45));
  float blueKey=pow(max(0.,dot(reflectedDirection,normalize(vec3(-.6,.35,1.)))),3.);
  reflection+=vec3(.018,.045,.30)*blueKey;
  vec3 halfVector=normalize(-incident+normalize(vec3(-1.,.8,1.)));
  float glint=pow(max(0.,dot(normalWorld,halfVector)),mix(240.,18.,uRoughness))*.6*(1.-uRoughness*.65);
  vec3 result=transmitted*1.35+reflection*(.12+fresnel*.32)+glint;
  result+=packets*(.75+fresnel*.85)+throughPackets*.35;
  vec3 pointerDirection=normalize(vec3(uPointerLight.x*5.,3.-uPointerLight.y*3.,7.)-vWorld);
  float pointerGlint=pow(max(0.,dot(normalWorld,normalize(pointerDirection-incident))),mix(85.,12.,uRoughness))*(1.-uRoughness*.65);
  result+=vec3(.16,.19,.25)*pointerGlint*uPointerLight.z;
  gl_FragColor=vec4(result,1.);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;
const sideFragment=`
uniform float uTime,uEnergy;
varying vec2 vUv;
void main(){
  vec2 uv=vUv;
  float bands=.5+.5*sin(uv.y*17.+sin(uv.x*9.+uTime*.25)*2.);
  float sheen=pow(.5+.5*cos(uv.y*6.-uv.x*4.+uTime*.15),4.);
  vec3 color=mix(vec3(.015,.025,.12),vec3(.1,.15,.65),bands*.65);
  color+=vec3(.05,.09,.25)*sheen*(.2+uEnergy);
  gl_FragColor=vec4(color,.48);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;

export async function createHeroA(T,renderer,{invalidate=()=>{},onVideoStatus=()=>{}}={}) {
  const media=window.matchMedia('(prefers-reduced-motion: reduce)');
  const url=name=>new URL(`./assets/${name}`,import.meta.url).href;
  const [data,environment,poster]=await Promise.all([
    fetch(url('hero-a.geometry.json')).then(response=>{if(!response.ok)throw Error('Hero geometry unavailable');return response.json();}),
    new T.CubeTextureLoader().loadAsync(['px','nx','py','ny','pz','nz'].map(face=>url(`env-${face}.png`))),
    new T.TextureLoader().loadAsync(url('packet-reflection-poster.jpg')),
  ]);
  if(data.schema!=='argus/hero-reference-geometry/v1')throw Error('Unexpected hero asset');
  environment.colorSpace=T.SRGBColorSpace;
  poster.colorSpace=T.SRGBColorSpace;
  const group=new T.Group(),inner=new T.Group();group.add(inner);
  inner.scale.setScalar(8);inner.position.y=-.34565;
  const backdrop=new T.WebGLRenderTarget(1,1,{depthBuffer:true});
  const noiseCanvas=document.createElement('canvas');noiseCanvas.width=noiseCanvas.height=64;
  const noiseContext=noiseCanvas.getContext('2d'),pixels=noiseContext.createImageData(64,64);
  let seed=0x73636f70;
  for(let i=0;i<pixels.data.length;i+=4){seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;const n=(seed>>>0)%256;pixels.data.set([n,n,n,255],i);}
  noiseContext.putImageData(pixels,0,0);
  const noise=new T.CanvasTexture(noiseCanvas);noise.wrapS=noise.wrapT=T.RepeatWrapping;
  const uniforms={uBackdrop:{value:backdrop.texture},uPacketVideo:{value:poster},uDecodeVideo:{value:0},uRoughness:{value:DEFAULT_HERO_ROUGHNESS},uNoise:{value:noise},uEnvironment:{value:environment},uResolution:{value:new T.Vector2(1,1)},uViewToWorld:{value:new T.Matrix3()},uPointerLight:{value:new T.Vector3(0,0,0)}};
  const video=document.createElement('video');
  video.hidden=true;video.setAttribute('aria-hidden','true');video.dataset.role='hero-reflection';
  video.src=url('packet-reflection.mp4');renderer.domElement.parentElement.append(video);
  const videoTexture=new T.VideoTexture(video);videoTexture.colorSpace=T.SRGBColorSpace;
  let videoFrames=0;
  const playback=createHeroVideo(video,{
    invalidate,
    onFrame(){if(!media.matches&&document.documentElement.dataset.motion!=='reduced'){uniforms.uPacketVideo.value=videoTexture;uniforms.uDecodeVideo.value=1;}renderer.domElement.dataset.reflectionFrames=String(++videoFrames);},
    onStatus(status){if(status==='error'||status==='blocked'){uniforms.uPacketVideo.value=poster;uniforms.uDecodeVideo.value=0;}onVideoStatus(status);},
  });
  const glass=new T.ShaderMaterial({vertexShader:glassVertex,fragmentShader:glassFragment,uniforms});
  const outline=new T.MeshBasicMaterial({color:0xdde7ff,transparent:true,opacity:.18,depthWrite:false});
  const side=new T.ShaderMaterial({vertexShader:glassVertex,fragmentShader:sideFragment,uniforms:{uTime:{value:0},uEnergy:{value:0}},transparent:true,depthWrite:false,side:T.DoubleSide});
  const geometries=[];
  for(const name of ['Alche_A','Alche_Outline','Alche_SideScreen']) {
    const meshData=data.meshes.find(item=>item.name===name);
    if(!meshData)throw Error(`Hero mesh missing: ${name}`);
    const geometry=new T.BufferGeometry();
    geometry.setAttribute('position',new T.BufferAttribute(decodeHeroStream(meshData.position),3));
    if(meshData.normal)geometry.setAttribute('normal',new T.BufferAttribute(decodeHeroStream(meshData.normal),3));
    if(meshData.uv)geometry.setAttribute('uv',new T.BufferAttribute(decodeHeroStream(meshData.uv),2));
    const indices=decodeHeroStream(meshData.index);
    if(indices.some(index=>index>=meshData.position.count))throw Error('Hero index exceeds vertex bounds');
    geometry.setIndex(new T.BufferAttribute(indices,1));
    geometries.push(geometry);
    const mesh=new T.Mesh(geometry,name==='Alche_A'?glass:name==='Alche_Outline'?outline:side);
    mesh.name=name;mesh.renderOrder=name==='Alche_A'?1:name==='Alche_Outline'?2:3;
    inner.add(mesh);
  }
  // Render the actual curved studio and its lettering in this GL context.
  // No DOM pixels or separately positioned flat wordmark copy are sampled.
  let motion={x:0,y:0,vx:0,vy:0,active:false},lastPointer=null,lastTime=0;
  let enabled=false;
  function syncPlayback() {
    const reduced=media.matches||document.documentElement.dataset.motion==='reduced';
    if(reduced){uniforms.uPacketVideo.value=poster;uniforms.uDecodeVideo.value=0;}
    playback.setEnabled(enabled&&!reduced);
    invalidate();
  }
  media.addEventListener('change',syncPlayback);
  const motionObserver=new MutationObserver(syncPlayback);
  motionObserver.observe(document.documentElement,{attributes:true,attributeFilter:['data-motion']});
  function resize(w,h) {
    const size=renderer.getDrawingBufferSize(new T.Vector2());
    backdrop.setSize(size.x,size.y);uniforms.uResolution.value.copy(size);
  }
  function hover(x,y) {
    const prior=lastPointer;lastPointer={x,y};
    if(!prior||media.matches||document.documentElement.dataset.motion==='reduced')return;
    const influence=Math.max(0,1-Math.hypot(x,y)*1.1);
    motion.vy=Math.max(-2,Math.min(2,motion.vy+(x-prior.x)*4*influence));
    motion.vx=Math.max(-1.5,Math.min(1.5,motion.vx+(y-prior.y)*3*influence));
    motion.active=true;
  }
  function tick(time) {
    const delta=lastTime?Math.min(.033,(time-lastTime)/1000):.033;lastTime=time;
    motion=stepHeroMotion(motion,delta,media.matches||document.documentElement.dataset.motion==='reduced');
    inner.rotation.set(motion.x,motion.y,0);
    side.uniforms.uTime.value=time/1000;side.uniforms.uEnergy.value=Math.min(1,Math.abs(motion.vx)+Math.abs(motion.vy));
    return motion.active||playback.needsPaintLoop;
  }
  function prepare(scene,camera) {
    const prior=renderer.getRenderTarget(),clear=new T.Color();renderer.getClearColor(clear);const alpha=renderer.getClearAlpha();
    group.visible=false;
    renderer.setRenderTarget(backdrop);renderer.setClearColor(0x050608,1);renderer.clear();
    const auto=renderer.autoClear;renderer.autoClear=false;
    renderer.render(scene,camera);
    renderer.autoClear=auto;renderer.setRenderTarget(prior);renderer.setClearColor(clear,alpha);group.visible=true;
    uniforms.uViewToWorld.value.setFromMatrix4(camera.matrixWorld);
  }
  return {group,resize,hover,tick,prepare,leave(){lastPointer=null;},
    setRoughness(value){uniforms.uRoughness.value=clampHeroRoughness(value);renderer.domElement.dataset.roughness=uniforms.uRoughness.value.toFixed(2);invalidate();},
    setLight(value){uniforms.uPointerLight.value.copy(value);},
    setEnabled(value){if(enabled!==!!value){enabled=!!value;syncPlayback();}},
    retryVideo(){playback.retry();},
    dispose(){playback.dispose();videoTexture.dispose();poster.dispose();media.removeEventListener('change',syncPlayback);motionObserver.disconnect();for(const geometry of geometries)geometry.dispose();glass.dispose();outline.dispose();side.dispose();noise.dispose();environment.dispose();backdrop.dispose();},
  };
}
