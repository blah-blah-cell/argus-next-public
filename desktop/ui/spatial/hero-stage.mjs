import {roomWallPoint,stepStageMotion} from './stage-math.mjs?v=room-orbit-1';

const vertex=`
varying vec2 vUv;
varying vec3 vWorld,vNormal;
void main(){
  vUv=uv;vWorld=(modelMatrix*vec4(position,1.)).xyz;
  vNormal=normalize(mat3(modelMatrix)*normal);
  gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);
}`;
const wallFragment=`
uniform sampler2D uType;
uniform vec3 uPointer;
uniform vec2 uViewport;
uniform float uWordScale,uWordOpacity,uProgress,uTime,uMotion,uFocus;
varying vec2 vUv;
varying vec3 vWorld,vNormal;
void main(){
  vec3 lightPosition=vec3(-6.+uPointer.x*10.,4.-uPointer.y*4.,5.);
  vec3 direction=normalize(lightPosition-vWorld);
  float diffuse=max(0.,dot(normalize(vNormal),direction));
  vec2 delta=vUv-vec2(.5+uPointer.x*.25,.48-uPointer.y*.18);
  float pool=exp(-dot(delta*vec2(4.,3.),delta*vec2(4.,3.)));
  float height=smoothstep(0.,.35,vUv.y);
  float edge=1.-smoothstep(.27,.52,abs(vUv.x-.5));
  vec3 wall=vec3(.007,.008,.010)+vec3(.012,.013,.016)*diffuse*height;
  wall+=vec3(.010,.012,.016)*pool*(.35+.65*uPointer.z);
  wall*=.72+.28*edge;
  vec2 roomUv=vec2(fract(vUv.x*4.+.5),vUv.y);
  vec2 typeUv=vec2((roomUv.x-.5)/uWordScale+.5,(roomUv.y-.38+uProgress*.07)/(uWordScale*.95)+.5);
  // Distort only the sampled lettering: the wall, controls and hit targets stay put.
  vec2 screen=gl_FragCoord.xy/uViewport;
  vec2 pointer=vec2(.5+uPointer.x*.5,.5-uPointer.y*.5);
  vec2 cursorDelta=screen-pointer;
  vec2 metricDelta=cursorDelta*vec2(uViewport.x/uViewport.y,1.);
  float influence=exp(-dot(metricDelta,metricDelta)*75.);
  float warp=uMotion*uPointer.z*(1.-uFocus)*influence;
  // Quiet magnetic lens: one smooth pull, without waves or time-varying ripples.
  typeUv+=cursorDelta*vec2(.12,.12)*warp;
  // Rigid whole-word drift; every letter gets the same tiny offset.
  typeUv+=vec2(sin(uTime*.18),cos(uTime*.14))*.0015*uMotion;
  float inside=step(0.,typeUv.x)*step(typeUv.x,1.)*step(0.,typeUv.y)*step(typeUv.y,1.);
  float ink=texture2D(uType,clamp(typeUv,0.,1.)).r*inside;
  float offsetInk=texture2D(uType,clamp(typeUv+vec2(-.0018,.0032),0.,1.)).r*inside;
  float raisedEdge=max(ink-offsetInk,0.);
  float castShadow=max(offsetInk-ink,0.);
  vec3 lettering=vec3(.38,.42,.48)*(.7+.3*diffuse)+vec3(.25,.27,.3)*pool;
  vec3 color=wall*(1.-castShadow*.65*uWordOpacity);
  color=mix(color,lettering,ink*uWordOpacity);
  color+=raisedEdge*vec3(.21,.24,.3)*uWordOpacity;
  gl_FragColor=vec4(color,1.);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;
const floorFragment=`
uniform vec3 uPointer;
varying vec3 vWorld;
void main(){
  vec2 p=vWorld.xz;
  float shadow=exp(-dot(p/vec2(2.,1.15),p/vec2(2.,1.15)));
  vec2 lightDelta=p-vec2(uPointer.x*5.,1.+uPointer.y*2.);
  float pool=exp(-dot(lightDelta/vec2(7.,5.),lightDelta/vec2(7.,5.)));
  vec3 color=vec3(.006,.007,.009)+vec3(.008,.009,.012)*pool;
  color*=1.-shadow*.78;
  gl_FragColor=vec4(color,1.);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;

export function createHeroStage(T,host,canvas) {
  const group=new T.Group();group.name='argus-curved-studio';
  const typeCanvas=document.createElement('canvas');typeCanvas.width=2048;typeCanvas.height=1024;
  const context=typeCanvas.getContext('2d');
  const word=document.querySelector('.spatial-wordmark');
  const family=getComputedStyle(word).fontFamily;
  context.fillStyle='#000';context.fillRect(0,0,2048,1024);
  context.fillStyle='#fff';context.font=`800 460px ${family}`;context.textBaseline='alphabetic';
  const letters=[...word.textContent.trim()],tracking=-22;
  const widths=letters.map(letter=>context.measureText(letter).width);
  let left=(2048-widths.reduce((sum,width)=>sum+width,0)-tracking*(letters.length-1))/2;
  for(let i=0;i<letters.length;i++){context.fillText(letters[i],left,674);left+=widths[i]+tracking;}
  const typeTexture=new T.CanvasTexture(typeCanvas);typeTexture.colorSpace=T.NoColorSpace;
  const pointerLight=new T.Vector3(0,0,0);
  const uniforms={uType:{value:typeTexture},uPointer:{value:pointerLight},uViewport:{value:new T.Vector2(1,1)},uWordScale:{value:.8},uWordOpacity:{value:1},uProgress:{value:0},uTime:{value:0},uMotion:{value:0},uFocus:{value:0}};
  const wallGeometry=new T.PlaneGeometry(1,1,192,8);
  const positions=wallGeometry.attributes.position;
  for(let i=0;i<positions.count;i++)positions.setXYZ(i,...roomWallPoint(positions.getX(i)+.5,positions.getY(i)+.5));
  positions.needsUpdate=true;wallGeometry.computeVertexNormals();
  const wallMaterial=new T.ShaderMaterial({vertexShader:vertex,fragmentShader:wallFragment,uniforms,side:T.DoubleSide});
  const wall=new T.Mesh(wallGeometry,wallMaterial);wall.name='curved-letter-wall';group.add(wall);
  const floorGeometry=new T.PlaneGeometry(100,60);
  const floorMaterial=new T.ShaderMaterial({vertexShader:vertex,fragmentShader:floorFragment,uniforms:{uPointer:{value:pointerLight}}});
  const floor=new T.Mesh(floorGeometry,floorMaterial);floor.rotation.x=-Math.PI/2;floor.position.y=-2.17;floor.position.z=5;group.add(floor);
  const cursor=document.createElement('div');cursor.className='spatial-pointer';cursor.hidden=true;cursor.setAttribute('aria-hidden','true');host.append(cursor);
  const media=window.matchMedia('(prefers-reduced-motion: reduce)');
  const fine=window.matchMedia('(hover: hover) and (pointer: fine)');
  let state={x:0,y:0,alpha:0,focus:0,active:false},target={x:0,y:0,alpha:0,focus:0};
  let width=1,height=1,lastTime=0,phase=0,enabled=false;
  const reduced=()=>media.matches||document.documentElement.dataset.motion==='reduced';
  return {
    group,pointerLight,
    resize(w,h) {width=w;height=h;uniforms.uViewport.value.set(canvas.width||w,canvas.height||h);uniforms.uWordScale.value=Math.max(.28,Math.min(.94,w/h*.51));},
    pointer(x,y,interactive=false) {
      if(!enabled||!fine.matches||reduced())return;
      target={x,y,alpha:1,focus:interactive?1:0};
    },
    leave(){target={x:0,y:0,alpha:0,focus:0};},
    setEnabled(value){enabled=value;if(!value){target={x:0,y:0,alpha:0,focus:0};state={...target,active:false};pointerLight.set(0,0,0);cursor.hidden=true;uniforms.uMotion.value=0;canvas.dataset.textMotion='static';lastTime=0;}},
    tick(time,progress,cameraYaw=0) {
      const delta=lastTime?Math.max(0,Math.min(.05,(time-lastTime)/1000)):1/30;lastTime=time;
      state=stepStageMotion(state,target,delta,reduced()||!fine.matches);
      pointerLight.set(state.x,state.y,state.alpha);
      const t=Math.max(0,Math.min(1,(progress-.1)/.34));
      uniforms.uWordOpacity.value=1-t*.92;uniforms.uProgress.value=t;
      const animate=enabled&&!reduced()&&fine.matches&&uniforms.uWordOpacity.value>.001;
      if(animate)phase=(phase+delta)%10000;
      uniforms.uMotion.value=animate?1:0;uniforms.uTime.value=phase;uniforms.uFocus.value=state.focus;
      // Do not follow cameraYaw: that cancels the room's apparent rotation.
      group.rotation.y=state.x*.005*state.alpha;
      cursor.hidden=!enabled||state.alpha<.002||reduced()||!fine.matches;
      cursor.style.opacity=String(state.alpha*(.65-.2*state.focus));
      cursor.style.transform=`translate3d(${(state.x+1)*width/2}px,${(state.y+1)*height/2}px,0) translate(-50%,-50%) scale(${1-state.focus*.38})`;
      canvas.dataset.backdrop='curved-wall';canvas.dataset.pointer=`${state.x.toFixed(3)},${state.y.toFixed(3)},${state.alpha.toFixed(3)}`;
      canvas.dataset.textMotion=animate?'active':'static';canvas.dataset.textPhase=phase.toFixed(3);
      return state.active||animate;
    },
    dispose(){cursor.remove();typeTexture.dispose();wallGeometry.dispose();wallMaterial.dispose();floorGeometry.dispose();floorMaterial.dispose();},
  };
}
