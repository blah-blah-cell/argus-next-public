// Native timeline owns progress. Scroll events wake the renderer; they do not
// measure layout, change document scrolling, or compute camera positions.
export function createScrollStory(subject, wake, {onModeChange}={}) {
  const media=window.matchMedia('(prefers-reduced-motion: reduce)');
  const probe=document.getElementById('spatial-progress');
  let animation=null, destroyed=false;
  function configure() {
    animation?.cancel();animation=null;
    const supported=typeof ViewTimeline==='function';
    const reduced=media.matches||document.documentElement.dataset.motion==='reduced';
    subject.classList.toggle('scroll-story',supported&&!reduced);
    subject.dataset.scroll=supported?(reduced?'reduced':'native-timeline'):'static-fallback';
    onModeChange?.(supported&&!reduced);
    if(supported&&!reduced) {
      animation=probe.animate([{opacity:0},{opacity:1}],{
        timeline:new ViewTimeline({subject,axis:'block'}), duration:1,fill:'both',
        rangeStart:'contain 0%',rangeEnd:'contain 100%',easing:'linear',
      });
    }
    wake();
  }
  const observer=new MutationObserver(configure);
  observer.observe(document.documentElement,{attributes:true,attributeFilter:['data-motion']});
  media.addEventListener('change',configure);
  const notify=()=>{if(animation&&!destroyed)wake();};
  document.addEventListener('scroll',notify,{passive:true});
  document.addEventListener('scrollend',notify,{passive:true});
  configure();
  return {
    progress() { return animation?.effect.getComputedTiming().progress??0; },
    animate(target,keyframes,options={}) {
      if(!animation)return null;
      return target.animate(keyframes,{duration:1,fill:'both',timeline:animation.timeline,rangeStart:'contain 0%',rangeEnd:'contain 100%',...options});
    },
    destroy() {
      destroyed=true;animation?.cancel();observer.disconnect();
      media.removeEventListener('change',configure);
      document.removeEventListener('scroll',notify);document.removeEventListener('scrollend',notify);
    },
  };
}
