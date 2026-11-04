export const DEFAULT_HERO_ROUGHNESS=.18;
export function clampHeroRoughness(value) {
  return Number.isFinite(value)?Math.max(0,Math.min(1,value)):DEFAULT_HERO_ROUGHNESS;
}

export function decodeHeroStream(stream) {
  const Type=stream?.componentType===5126?Float32Array:stream?.componentType===5123?Uint16Array:null;
  if(!Type||!Number.isInteger(stream.count)||stream.count<1||stream.count>100000||![1,2,3].includes(stream.components))throw Error('Invalid hero accessor');
  const length=stream.count*stream.components*Type.BYTES_PER_ELEMENT;
  if(typeof stream.data!=='string'||stream.data.length>2_000_000)throw Error('Invalid hero buffer');
  const raw=atob(stream.data);
  if(raw.length!==length)throw Error('Hero buffer length mismatch');
  const bytes=Uint8Array.from(raw,char=>char.charCodeAt(0)),values=new Type(bytes.buffer);
  if(!values.every(Number.isFinite))throw Error('Non-finite hero geometry');
  return values;
}

export function stepHeroMotion(state, delta, reduced=false) {
  if(reduced)return {x:0,y:0,vx:0,vy:0,active:false};
  const dt=Math.max(0,Math.min(1/30,Number.isFinite(delta)?delta:0));
  const finite=value=>Number.isFinite(value)?value:0;
  let x=finite(state.x),y=finite(state.y),vx=finite(state.vx),vy=finite(state.vy);
  vx+=(-12*x-6*vx)*dt;vy+=(-12*y-6*vy)*dt;
  x=Math.max(-.45,Math.min(.45,x+vx*dt));y=Math.max(-.6,Math.min(.6,y+vy*dt));
  const active=Math.abs(x)+Math.abs(y)+Math.abs(vx)+Math.abs(vy)>.0008;
  return active?{x,y,vx,vy,active}:{x:0,y:0,vx:0,vy:0,active:false};
}
