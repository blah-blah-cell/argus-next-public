import {roomCamera} from './room-motion.mjs?v=room-orbit-1';
const clamp=value=>Number.isFinite(value)?Math.max(0,Math.min(1,value)):0;
export const setupProgress=value=>.48+.52*value;
export const SETUP_STOPS=Object.freeze([.28,.59,.90].map(setupProgress));
export const INTRO_STOPS=Object.freeze([.10,.25,.41]);
export const introChapter=value=>value<.17?0:value<.33?1:2;
export function setupStep(progress) {
  const p=(clamp(progress)-.48)/.52;return p<.14?-1:p<.44?0:p<.74?1:2;
}
const frames=[
  [0, {x:0,y:.12,rotationY:-.24,rotationZ:0,scale:1.35,worldYaw:0,worldPitch:0,worldRoll:0,cameraYaw:0,cameraPitch:0,cameraRoll:0}],
  [.22,{x:-2.35,y:.15,rotationY:.45,rotationZ:-.035,scale:1.03,worldYaw:-.12,worldPitch:.015,worldRoll:.025,cameraYaw:.1,cameraPitch:.02,cameraRoll:.012}],
  [.37,{x:-2.35,y:.15,rotationY:.45,rotationZ:-.035,scale:1.03,worldYaw:-.12,worldPitch:.015,worldRoll:.025,cameraYaw:.1,cameraPitch:.02,cameraRoll:.012}],
  [.51,{x:-2.35,y:.18,rotationY:2.35,rotationZ:.025,scale:1.02,worldYaw:.18,worldPitch:-.025,worldRoll:-.025,cameraYaw:-.1,cameraPitch:.08,cameraRoll:-.012}],
  [.67,{x:-2.35,y:.18,rotationY:2.35,rotationZ:.025,scale:1.02,worldYaw:.18,worldPitch:-.025,worldRoll:-.025,cameraYaw:-.1,cameraPitch:.08,cameraRoll:-.012}],
  [.81,{x:-2.35,y:.12,rotationY:6.5,rotationZ:-.02,scale:1.03,worldYaw:-.15,worldPitch:.02,worldRoll:.02,cameraYaw:.1,cameraPitch:-.02,cameraRoll:.012}],
  [1, {x:-2.35,y:.12,rotationY:6.5,rotationZ:-.02,scale:1.03,worldYaw:-.15,worldPitch:.02,worldRoll:.02,cameraYaw:.1,cameraPitch:-.02,cameraRoll:.012}],
];
const introPose=(rotationY,worldYaw,cameraYaw)=>({...frames[1][1],rotationY,worldYaw,cameraYaw});
const storyFrames=[
  frames[0],
  [.08,introPose(-.24,-.08,.06)],[.14,introPose(-.24,-.08,.06)],
  [.22,introPose(-1.4,.14,-.08)],[.30,introPose(-1.4,.14,-.08)],
  [.38,introPose(-2.4,-.12,.08)],[.46,introPose(-2.4,-.12,.08)],
  ...frames.map(([p,value],index)=>[setupProgress(p),index===0?introPose(-2.4,-.12,.08):value]),
];
export function journeyPose(value) {
  const p=clamp(value);let index=1;while(index<storyFrames.length-1&&p>storyFrames[index][0])index++;
  const [from,a]=storyFrames[index-1],[to,b]=storyFrames[index];
  const t=clamp((p-from)/(to-from)),s=t*t*(3-2*t);
  const step=setupStep(p);
  const pose={progress:p,chapter:step<0?['identity','memory','evidence'][introChapter(p)]:['privacy','checks','start'][step]};
  for(const key of Object.keys(a))pose[key]=s===0?a[key]:s===1?b[key]:a[key]+(b[key]-a[key])*s;
  Object.assign(pose,roomCamera(p));
  return pose;
}
