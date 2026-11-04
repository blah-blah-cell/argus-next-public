const INTRO_KEY='argus.intro.seen.v1';
const SETUP_KEY='argus-onboarding-complete';
export function createIntroSession(storage) {
  let eligible=false;
  try { eligible=storage?.getItem(INTRO_KEY)!=='1'&&storage?.getItem(SETUP_KEY)!=='argus-onboarding/v1'; } catch { /* Replay remains available without storage. */ }
  return {
    takeFirstRun() { const result=eligible;eligible=false;return result; },
    dismiss() { eligible=false;try{storage?.setItem(INTRO_KEY,'1');}catch{ /* Session-only dismissal. */ } },
  };
}
