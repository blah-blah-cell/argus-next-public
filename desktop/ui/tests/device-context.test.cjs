const {test} = require('node:test');
const assert = require('node:assert/strict');
const load = () => import('../device-context.mjs');
test('onboarding restores a real connection mode without reading Defender automatically', async () => {
  const {mountOnboardingDefender} = await load();
  let click, calls = 0;
  const button = {disabled:false,addEventListener:(_,fn)=>{click=fn;}};
  const status = {setAttribute() {}};
  const mode = {addEventListener() {}};
  const elements = {'onboarding-defender-connect':button,'onboarding-defender-status':status,'onboarding-defender-mode':mode};
  mountOnboardingDefender({getElementById:id=>elements[id]}, async command => {
    if(command==='runtime_preferences')return {defenderMode:'read_only'};
    calls++; assert.equal(command,'local_device_context');
    return {schema:'argus.device-context/v1',connections:[],detections:[],observedAt:'now',defender:{antivirusEnabled:true,realTimeProtectionEnabled:false},detectionError:'Access denied'};
  });
  await Promise.resolve();
  assert.equal(mode.value,'read_only');
  assert.equal(calls,0);
  await click();
  assert.match(status.textContent,/real-time protection: off/);
  assert.match(status.textContent,/history is unavailable/);
  assert.equal(button.disabled,false);
});
test('onboarding Defender provider failures do not prevent retry or finish setup', async () => {
  const {mountOnboardingDefender} = await load();
  let click;
  const button = {disabled:false,addEventListener:(_,fn)=>{click=fn;}};
  const status = {setAttribute() {}};
  mountOnboardingDefender({getElementById:id=>({'onboarding-defender-connect':button,'onboarding-defender-status':status})[id]}, async command => command==='runtime_preferences'?{defenderMode:'read_only'}:({schema:'argus.device-context/v1',connections:[],detections:[],defender:null,defenderError:'Access denied'}));
  await Promise.resolve();
  await click();
  assert.match(status.textContent,/Access denied/);
  assert.match(status.textContent,/finish setup/);
  assert.equal(button.disabled,false);
});
test('onboarding disconnect is persisted and a failed mode change restores the prior value', async () => {
  const {mountOnboardingDefender} = await load();
  let change, fail = false;
  const mode = {value:'',disabled:false,addEventListener:(_,fn)=>{change=fn;}};
  const button = {addEventListener() {}};
  const status = {setAttribute() {}};
  const elements = {'onboarding-defender-mode':mode,'onboarding-defender-connect':button,'onboarding-defender-status':status};
  mountOnboardingDefender({getElementById:id=>elements[id]}, async (command,args) => {if(command==='runtime_preferences')return {defenderMode:'read_only'}; if(fail)throw Error('Disk full');return {defenderMode:args.mode};});
  await Promise.resolve();
  mode.value='off'; await change(); assert.equal(button.disabled,true); assert.match(status.textContent,/Disconnected/);
  fail=true; mode.value='controls'; await change(); assert.equal(mode.value,'off'); assert.equal(mode.disabled,false); assert.match(status.textContent,/not saved/);
});
test('Defender control cancellation never invokes an elevated action', async () => {
  const {mountDefenderControls} = await load();
  let click, close, calls=0;
  const button = {dataset:{defenderAction:'quick_scan'},addEventListener:(_,fn)=>{click=fn;},focus(){}};
  const dialog = {open:false,showModal(){this.open=true;},querySelector(){return {focus(){}};},addEventListener:(_,fn)=>{close=fn;}};
  const elements = {'defender-action-confirm':dialog,'defender-action-status':{},'defender-confirm-action':{}};
  mountDefenderControls({getElementById:id=>elements[id],querySelectorAll:()=>[button]},async()=>{calls++;});
  click(); assert.equal(calls,0); dialog.returnValue='cancel'; await close(); assert.equal(calls,0);
  dialog.open=false; click(); dialog.returnValue='confirm'; await close(); assert.equal(calls,1); assert.equal(button.disabled,false);
});
test('device context validates schema and preserves partial failures', async () => {
  const {deviceSummary} = await load();
  assert.throws(() => deviceSummary({}));
  assert.throws(() => deviceSummary({schema: 'argus.device-context/v1', connections: {}}));
  const data = {schema: 'argus.device-context/v1', connections: [], detections: [], defenderError: 'Access denied'};
  assert.equal(deviceSummary(data), data);
});
test('device context is on demand and recovers from runtime errors', async () => {
  const {mountDeviceContext} = await load();
  let click, calls = 0;
  const elements = {
    'device-context-refresh': {disabled: false, addEventListener: (_, fn) => {click = fn;}},
    'device-context-status': {},
    'device-context-results': {replaceChildren() {}, setAttribute() {}},
  };
  mountDeviceContext({getElementById: id => elements[id]}, async command => { calls++; assert.equal(command, 'local_device_context'); throw Error('Access denied'); });
  assert.equal(calls, 0);
  await click();
  assert.equal(calls, 1);
  assert.match(elements['device-context-status'].textContent, /Access denied/);
  assert.equal(elements['device-context-refresh'].disabled, false);
});
test('renders provider data as text without interpreting process names as markup', async () => {
  const {mountDeviceContext} = await load();
  let click;
  const nodes = [];
  const element = () => ({children: [], textContent: '', append(...items) {this.children.push(...items);}, setAttribute() {}, replaceChildren() {this.children = [];}});
  const button = {...element(), disabled: false, addEventListener: (_, fn) => {click = fn;}};
  const status = element(), results = element();
  const doc = {getElementById: id => ({'device-context-refresh': button, 'device-context-status': status, 'device-context-results': results})[id], createElement: () => { const n = {...element(), style: {}}; nodes.push(n); return n; }};
  mountDeviceContext(doc, async () => ({schema: 'argus.device-context/v1', observedAt: 'now', defender: null, defenderError: 'Unavailable', detectionError: null, detections: [], connections: [{processName: '<img onerror=alert(1)>', processId: 42, protocol: 'UDP', localAddress: '::', localPort: 53, state: 'Bound endpoint'}]}));
  await click();
  assert(nodes.some(n => n.textContent === '<img onerror=alert(1)> / 42'));
  assert(nodes.some(n => n.textContent === 'Not available'));
  assert(nodes.every(n => !Object.hasOwn(n, 'innerHTML')));
  assert.equal(button.disabled, false);
});
test('automatic refresh is opt-in, survives frequent flow updates and stops with monitoring', async t => {
  t.mock.timers.enable({apis:['setTimeout']});
  const {mountDeviceContext} = await load();
  const handlers = {};
  const button = {disabled:false,addEventListener() {}};
  const auto = {checked:false,addEventListener(_,fn) {this.change = fn;}};
  const elements = {'device-context-refresh':button,'device-context-auto':auto,'device-context-status':{},'device-context-results':{replaceChildren() {},setAttribute() {}}};
  let calls = 0;
  mountDeviceContext({defaultView:{addEventListener:(name,fn)=>{handlers[name]=fn;}},visibilityState:'visible',getElementById:id=>elements[id]}, async () => {calls++; throw Error('Unavailable');});
  handlers['argus:device-monitor']({detail:true});
  handlers['argus:device-flows']({detail:{live:true,flows:[]}});
  t.mock.timers.tick(20000); assert.equal(calls,0);
  auto.checked = true; auto.change(); assert.equal(calls,1);
  await Promise.resolve(); await Promise.resolve();
  t.mock.timers.tick(5000);
  handlers['argus:device-flows']({detail:{live:true,flows:[]}});
  t.mock.timers.tick(5000); assert.equal(calls,2);
  await Promise.resolve(); await Promise.resolve();
  handlers['argus:device-monitor']({detail:false});
  t.mock.timers.tick(30000); assert.equal(calls,2);
});
