const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const read=file=>fs.readFileSync(path.join(__dirname,'..',file),'utf8');
const source=read('app.js');
function extract(name){const start=source.indexOf(`function ${name}(`);const end=source.indexOf('\n}',start)+2;return source.slice(start,end);}
test('first-run boot suppresses workspace flash, while completed users are not intercepted',()=>{
  function run(value,blocked=false){const classes=new Set();vm.runInNewContext(read('onboarding-boot.js'),{localStorage:{getItem(){if(blocked)throw Error('blocked');return value;}},document:{documentElement:{classList:{add:name=>classes.add(name)}}}});return classes;}
  assert.equal(run(null).has('journey-pending'),true);
  assert.equal(run('argus-onboarding/v1').size,0);
  assert.equal(run(null,true).has('journey-pending'),true);
});
test('setup opens the scroll journey immediately without a modal or premature readiness claim',()=>{
  const events=[],state={};let resolve;
  const context=vm.createContext({state,Promise,onboardingPromise:null,onboardingCompleted:()=>false,setSettingsOpen(){},$:()=>({}),window:{dispatchEvent:event=>events.push(event.type)},CustomEvent:class{constructor(type){this.type=type;}}});
  vm.runInContext(extract('showOnboarding'),context);
  const first=context.showOnboarding();assert.equal(events[0],'argus:journey-open');
  assert.equal(context.showOnboarding(),first);resolve=state.onboardingResolve;resolve();
  assert.equal(events.length,1);
  assert.doesNotMatch(extract('showOnboarding'),/showModal|start_rolling_capture|setItem/);
  assert.ok(source.lastIndexOf('void showOnboarding();')<source.lastIndexOf('initialize().catch'));
});
test('completion marker is written only after successful native setup; failures stay inline',async()=>{
  function setup(fail){const calls=[],writes=[],nodes={};const state={rolling:false};const context=vm.createContext({state,ONBOARDING_VERSION:'argus-onboarding/v1',$:id=>nodes[id]??(nodes[id]={}),invoke:async name=>{calls.push(name);if(fail&&name==='start_rolling_capture')throw Error('permission denied');return {running:true,keepMonitoringAfterExit:false};},updateRollingControl(){state.rolling=true;},localStorage:{setItem:(...args)=>writes.push(args)},notice(){},closeOnboarding(){calls.push('closed');}});vm.runInContext('async '+extract('completeOnboarding'),context);return {context,state,calls,writes,nodes};}
  const failure=setup(true);await failure.context.completeOnboarding();
  assert.equal(failure.writes.length,0);assert.equal(failure.calls.includes('closed'),false);
  assert.match(failure.nodes['onboarding-finish-status'].textContent,/permission denied/);
  assert.equal(failure.nodes['onboarding-next'].disabled,false);
  const success=setup(false);await success.context.completeOnboarding();
  assert.deepEqual(success.calls,['set_keep_monitoring_after_exit','start_rolling_capture','closed']);
  assert.equal(success.writes[0][0],'argus-onboarding-complete');assert.equal(success.state.onboardingDoneThisSession,true);
});
test('journey setup shares the hero document, hides/inerts daily tools, and is reduced-motion safe',()=>{
  const html=read('index.html'),workspace=read('spatial/workspace.mjs'),css=read('spatial/journey.css');
  assert.match(workspace,/workspace.append\(setup\)/);
  assert.match(workspace,/createScrollStory\(visualStory/);
  assert.match(workspace,/if\(window.ArgusSetup\?\.pending\)void openIntroduction/);
  assert.match(workspace,/child.inert=true/);assert.match(workspace,/child.inert=false/);
  assert.match(workspace,/scene\?\.setEnabled\(introOpen&&target==='overview'&&!flat\)/);
  assert.match(css,/\.journey-active \.spatial-pin\{height:100svh/);
  assert.match(css,/prefers-reduced-motion:reduce/);
  assert.match(css,/max-height:none/);assert.match(css,/overflow:visible/);
  assert.match(html,/<section\s+class="onboarding-dialog"/);
  assert.doesNotMatch(html,/<dialog\s+class="onboarding-dialog"/);
  assert.match(source,/\$\("spatial-workspace"\)\?\.scrollIntoView/);
});
test('completed setup bypasses first-run and blocked storage respects session completion',()=>{
  const context=vm.createContext({state:{},ONBOARDING_VERSION:'argus-onboarding/v1',localStorage:{getItem:()=> 'argus-onboarding/v1'}});
  vm.runInContext(extract('onboardingCompleted'),context);assert.equal(context.onboardingCompleted(),true);
  context.localStorage.getItem=()=>{throw Error('blocked');};assert.equal(context.onboardingCompleted(),false);
  context.state.onboardingDoneThisSession=true;assert.equal(context.onboardingCompleted(),true);
});
