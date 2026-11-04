import {clampHeroRoughness,DEFAULT_HERO_ROUGHNESS} from './hero-math.mjs?v=roughness-1';

export const ROUGHNESS_KEY='argus.hero.roughness.v1';
export function bindHeroRoughness(input,output,apply,storage) {
  let saved=null;
  try {saved=storage?.getItem(ROUGHNESS_KEY);}catch { /* Storage can be blocked. */ }
  let value=saved===null||String(saved).trim()===''?DEFAULT_HERO_ROUGHNESS:clampHeroRoughness(Number(saved)/100);
  function render() {
    const percent=Math.round(value*100);
    input.value=String(percent);output.textContent=`${percent}%`;
    input.setAttribute('aria-valuetext',`${percent}% roughness; ${percent===0?'polished':percent===100?'frosted':'glass'}`);
    apply(value);
  }
  function change() {
    value=clampHeroRoughness(Number(input.value)/100);render();
    try {storage?.setItem(ROUGHNESS_KEY,input.value);}catch { /* Still applies for this session. */ }
  }
  render();input.addEventListener('input',change);
  return {
    setEnabled(enabled){input.disabled=!enabled;input.title=enabled?'Polished to frosted glass':'Roughness is available in 3D view';},
    dispose(){input.removeEventListener('input',change);},
  };
}
