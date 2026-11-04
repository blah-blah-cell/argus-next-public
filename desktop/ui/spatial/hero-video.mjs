// Playback ownership only. The caller supplies visibility/reduced-motion policy.
// There is no independent animation timer: decoded frames wake the scene.
export function createHeroVideo(video,{onFrame=()=>{},onStatus=()=>{},invalidate=()=>{}}={}) {
  let enabled=false,disposed=false,failed=false,blocked=false,request=null,attempt=0,status='';
  const supportsFrames=typeof video.requestVideoFrameCallback==='function';
  function emit(next) {if(status!==next){status=next;onStatus(next);}}
  function cancel() {
    if(request!==null)video.cancelVideoFrameCallback?.(request);
    request=null;
  }
  function ready() {
    if(disposed||failed||video.readyState<2)return;
    onFrame();invalidate();
  }
  function queue() {
    if(disposed||failed||!enabled||video.paused||!supportsFrames||request!==null)return;
    request=video.requestVideoFrameCallback(()=> {
      request=null;
      if(disposed||failed||!enabled||video.paused)return;
      ready();queue();
    });
  }
  async function start() {
    if(disposed||failed||blocked||!enabled)return;
    const token=++attempt;
    try {
      await video.play();
      if(disposed||!enabled){video.pause();return;}
      if(token!==attempt)return;
      emit('playing');queue();invalidate();
    } catch {
      if(disposed||!enabled||token!==attempt)return;
      blocked=true;video.pause();cancel();emit('blocked');invalidate();
    }
  }
  function error() {
    if(disposed)return;
    failed=true;enabled=false;++attempt;video.pause();cancel();emit('error');invalidate();
  }
  function playing() {if(enabled&&!disposed){emit('playing');queue();}}
  const events={loadeddata:ready,seeked:ready,timeupdate:ready,playing,error};
  for(const [name,handler] of Object.entries(events))video.addEventListener(name,handler);
  video.muted=true;video.defaultMuted=true;video.loop=true;video.playsInline=true;video.preload='none';
  video.setAttribute('muted','');video.setAttribute('playsinline','');
  emit('poster');
  return {
    setEnabled(value) {
      const next=!!value&&!disposed&&!failed;
      if(next===enabled)return;
      enabled=next;
      if(enabled)start();
      else {++attempt;video.pause();cancel();emit(failed?'error':blocked?'blocked':'paused');}
    },
    retry() {if(blocked&&!disposed&&!failed){blocked=false;if(enabled)start();}},
    get needsPaintLoop() {return enabled&&!failed&&!blocked&&!video.paused&&!supportsFrames;},
    dispose() {
      disposed=true;enabled=false;++attempt;cancel();
      for(const [name,handler] of Object.entries(events))video.removeEventListener(name,handler);
      video.pause();video.removeAttribute('src');video.load();video.remove();
    },
  };
}
