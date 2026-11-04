// Avoid flashing the daily workspace on first launch.
try {
  if(localStorage.getItem('argus-onboarding-complete')!=='argus-onboarding/v1'){
    document.documentElement.classList.add('journey-pending');
    if(typeof history!=='undefined'&&'scrollRestoration' in history)history.scrollRestoration='manual';
  }
} catch {
  document.documentElement.classList.add('journey-pending');
  if(typeof history!=='undefined'&&'scrollRestoration' in history)history.scrollRestoration='manual';
}
