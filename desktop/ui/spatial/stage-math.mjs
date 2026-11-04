const finite=(value,fallback=0)=>Number.isFinite(value)?value:fallback;
const clamp=(value,low,high,fallback=0)=>Math.max(low,Math.min(high,finite(value,fallback)));

export function curveWallPoint(u,v) {
  const angle=(clamp(u,0,1,.5)-.5)*2;
  return [11*Math.sin(angle),-2.15+clamp(v,0,1,.5)*12,-6.2+11*(1-Math.cos(angle))];
}

// Camera stays inside this continuous panoramic wall; no front-facing overlay mesh.
export function roomWallPoint(u,v) {
  const angle=(clamp(u,0,1,.5)-.5)*Math.PI*2;
  return [18*Math.sin(angle),-2.15+clamp(v,0,1,.5)*14,-18*Math.cos(angle)];
}

export function stepStageMotion(state={},target={},delta=0,reduced=false) {
  if(reduced)return {x:0,y:0,alpha:0,focus:0,active:false};
  const dt=clamp(delta,0,.05),blend=1-Math.exp(-14*dt);
  const limits={x:[-1,1],y:[-1,1],alpha:[0,1],focus:[0,1]};
  const next={};let error=0;
  for(const [key,[low,high]] of Object.entries(limits)) {
    const end=clamp(target[key],low,high),start=clamp(state[key],low,high);
    const value=start+(end-start)*blend;
    next[key]=Math.abs(end-value)<.0005?end:value;
    error+=Math.abs(end-next[key]);
  }
  next.active=error>.001;
  if(!next.active)for(const [key,[low,high]] of Object.entries(limits))next[key]=clamp(target[key],low,high);
  return next;
}
