// One camera path for WebGL and the native-timeline DOM overlay. No DOM reads.
export const ROOM_STATIONS=Object.freeze([
  [0,0],[.08,.12],[.14,.12],[.22,.74],[.30,.74],
  [.38,1.40],[.48,1.40],[.5944,2.08],[.6724,2.08],
  [.7452,2.76],[.8284,2.76],[.9012,3.44],[1,3.44],
].map(frame=>Object.freeze(frame)));
export function roomCamera(value) {
  const p=Number.isFinite(value)?Math.max(0,Math.min(1,value)):0;
  let i=1;while(i<ROOM_STATIONS.length-1&&p>ROOM_STATIONS[i][0])i++;
  const [from,a]=ROOM_STATIONS[i-1],[to,b]=ROOM_STATIONS[i];
  const t=Math.max(0,Math.min(1,(p-from)/(to-from))),s=t*t*(3-2*t);
  const moving=a!==b,travel=moving&&t>0&&t<1?Math.sin(Math.PI*t)**2:0;
  return {cameraYaw:s===0?a:s===1?b:a+(b-a)*s,
    cameraPitch:.015+travel*.17,cameraRoll:travel===0?0:travel*-.075,
    cameraDolly:travel*1.65,travel};
}
export function roomUiFrames() {
  // Include exact hold boundaries in addition to samples; interpolate only transforms.
  const offsets=[...new Set([...ROOM_STATIONS.map(([p])=>p),...Array.from({length:201},(_,i)=>i/200)])].sort((a,b)=>a-b);
  return offsets.map(offset=>{
    const camera=roomCamera(offset),angle=camera.cameraRoll*180/Math.PI;
    return {offset,transform:`perspective(1600px) translate3d(0,${-camera.travel*22}px,${-camera.cameraDolly*75}px) rotateZ(${angle}deg) rotateX(${camera.travel*5}deg)`};
  });
}
