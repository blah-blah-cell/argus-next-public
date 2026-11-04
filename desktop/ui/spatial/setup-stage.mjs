import {SETUP_STOPS,INTRO_STOPS,introChapter,setupProgress,setupStep} from './journey-pose.mjs?v=room-orbit-1';
import {roomUiFrames} from './room-motion.mjs?v=room-orbit-1';

export function panelFrames(index) {
  const entry=[.14,.44,.74][index],start=[.22,.51,.81][index],end=[.37,.67,1][index],leave=[.44,.74,1][index];
  const incoming={opacity:0,transform:'perspective(1600px) translate3d(42vw,0,-380px) rotateY(-55deg)'};
  const held={opacity:1,transform:'perspective(1200px) translate3d(0,0,0) rotateY(0deg)'};
  const outgoing={opacity:0,transform:'perspective(1600px) translate3d(-42vw,0,-380px) rotateY(55deg)'};
  return [{...incoming,offset:0},{...incoming,offset:setupProgress(entry)},{...held,offset:setupProgress(start)},{...held,offset:setupProgress(end)},...(index<2?[{...outgoing,offset:setupProgress(leave)},{...outgoing,offset:1}]:[])];
}
export function createSetupStage({setup,pin,subject,workspace,onStep}) {
  const panels=[...setup.querySelectorAll('[data-onboarding-step]')];
  const footer=setup.querySelector('.onboarding-actions');
  const anchors=[...subject.querySelectorAll('[data-journey-anchor]')];
  const introductions=[...subject.querySelectorAll('[data-intro-copy]')];
  const introAnchors=[...subject.querySelectorAll('[data-intro-anchor]')];
  const narrative=subject.querySelector?.('.spatial-narrative');
  let cinematic=false,observer=null,animations=[],active=-1,driver=null,focusAfterScroll=null;
  function activate(step,chapter=introChapter(driver?.progress()??0)) {
    active=step;document.body.dataset.setupStage=step<0?'welcome':String(step);
    document.body.dataset.introChapter=String(chapter);
    introductions.forEach((panel,index)=>{panel.inert=cinematic&&(step>=0||index!==chapter);panel.setAttribute('aria-hidden',String(cinematic&&(step>=0||index!==chapter)));});
    panels.forEach((panel,index)=>{panel.inert=cinematic&&index!==step;panel.dataset.active=String(index===step);panel.setAttribute('aria-hidden',String(cinematic&&index!==step));});
    setup.querySelectorAll('[data-setup-target]').forEach(button=>{
      if(Number(button.dataset.setupTarget)===step)button.setAttribute('aria-current','step');else button.removeAttribute('aria-current');
    });
    onStep?.(step,chapter);
  }
  return {
    configure(enabled,nextDriver) {
      cinematic=enabled;driver=nextDriver;observer?.disconnect();animations.forEach(animation=>animation?.cancel());animations=[];
      document.body.classList.toggle('journey-cinematic',enabled);
      if(enabled) {
        pin.append(setup);panels[2].append(footer);
        panels.forEach((panel,index)=>animations.push(driver.animate(panel,panelFrames(index))));
        animations.push(driver.animate(setup,roomUiFrames()));
        if(narrative)animations.push(driver.animate(narrative,roomUiFrames()));
        observer=new IntersectionObserver(entries=>{
          const match=entries.find(entry=>entry.isIntersecting);
          if(match)activate(Number(match.target.dataset.journeyBand),Number(match.target.dataset.introChapter??2));
        },{rootMargin:'-49% 0px -49% 0px',threshold:0});
        subject.querySelectorAll('[data-journey-band]').forEach(band=>observer.observe(band));
        activate(setupStep(driver.progress()));
      } else {workspace.append(setup);setup.append(footer);activate(-1);}
    },
    next() {
      const progress=driver?.progress()??0;
      const nextIntro=INTRO_STOPS.findIndex(stop=>stop>progress+.02);
      if(cinematic&&nextIntro>=0)introAnchors[nextIntro].scrollIntoView({behavior:'smooth',block:'start'});
      else this.goToStep(active<0?0:Math.min(2,active+1));
    },
    goToStep(index) {
      const step=Math.max(0,Math.min(2,Number(index)||0));
      const target=cinematic?anchors[step]:panels[step];
      const reduced=document.documentElement.dataset.motion==='reduced'||window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      if(focusAfterScroll)document.removeEventListener('scrollend',focusAfterScroll);
      if(cinematic&&!reduced) {
        focusAfterScroll=()=>{if(active===step)panels[step].focus({preventScroll:true});};
        document.addEventListener('scrollend',focusAfterScroll,{once:true});
      }
      target.scrollIntoView({behavior:reduced?'instant':'smooth',block:'start'});
      if(!cinematic||reduced)panels[step].focus({preventScroll:true});
      if(step===1)onStep?.(step);
    },
    stop() {
      observer?.disconnect();animations.forEach(animation=>animation?.cancel());animations=[];
      if(focusAfterScroll)document.removeEventListener('scrollend',focusAfterScroll);
      document.body.classList.remove('journey-cinematic');
      panels.forEach(panel=>{panel.inert=false;panel.removeAttribute('aria-hidden');});
      introductions.forEach(panel=>{panel.inert=false;panel.removeAttribute('aria-hidden');});
    },
    get active(){return active;},
    stops:SETUP_STOPS,
  };
}
