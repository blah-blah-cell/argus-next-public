const {test,before}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
let controller,count,catalog,evidence,mount;
before(async()=>{({createLabController:controller,packetCount:count,scenarioCatalog:catalog,flowEvidence:evidence,mountSimulationLab:mount}=await import('../simulation-lab.mjs'));});
const scenarios=[{id:'baseline',title:'Baseline'},{id:'other',title:'Other'}];
const healthy=()=>({malformedRecords:0,reportedLostPackets:0,truncatedPackets:0,queueOverflows:0,parseErrors:0,sourceErrors:0});
const result=id=>({scenarioId:id,run:{health:healthy(),flows:[]},modelResult:{models:{test:{state:'forecast_only'}}}});
test('packet count strictly matches runtime bounds, not silent truncation',()=>{
  for(const value of [10,160,'500'])assert.equal(count(value),Number(value));
  for(const value of [null,'',9,501,10.5,Infinity,'invalid'])assert.throws(()=>count(value),/10 to 500/);
});
test('catalog includes every registered scenario and refuses duplicate or malformed IDs',()=>{
  assert.equal(catalog(scenarios).length,2);assert.equal(catalog(scenarios)[0].id,'baseline');
  assert.throws(()=>catalog(null));assert.throws(()=>catalog([...scenarios,scenarios[0]]));assert.throws(()=>catalog([{id:'bad'}]));
});
test('queue is serial, rejects overlap and stop waits for current then skips remaining',async()=>{
  let resolve,calls=0;const states=[];
  const lab=controller(id=>{calls++;return new Promise(r=>resolve=()=>r(result(id)));},state=>states.push(state));
  const first=lab.start(scenarios,160);
  assert.equal(lab.snapshot().busy,true);assert.equal(calls,1);
  assert.equal(await lab.start(scenarios,160),false);
  lab.stop();assert.equal(lab.snapshot().stopping,true);assert.equal(lab.snapshot().busy,true);
  resolve();await first;
  assert.equal(calls,1);assert.equal(lab.snapshot().busy,false);
  assert.deepEqual(lab.snapshot().queue.map(item=>item.status),['complete','skipped']);
  assert.equal(lab.snapshot().history[0].id,'baseline');
  assert.ok(states.some(state=>state.busy&&state.stopping));
});
test('failed or mismatched results are visible, do not claim success, and suite continues',async()=>{
  let calls=0;const lab=controller(async id=>{calls++;if(id==='baseline')throw Error('worker failed');return result(id);});
  await lab.start(scenarios,100);assert.equal(calls,2);
  assert.deepEqual(lab.snapshot().queue.map(item=>item.status),['failed','complete']);
  assert.match(lab.snapshot().history[1].error,/worker failed/);
  const mismatched=controller(async()=>result('invented'));await mismatched.start([scenarios[0]],160);
  assert.equal(mismatched.snapshot().history[0].status,'failed');
});
test('model errors remain partial outcomes, and history is bounded to ten runs',async()=>{
  const lab=controller(async id=>({...result(id),modelResult:null,modelError:'not installed'}));
  for(let i=0;i<12;i++)await lab.start([scenarios[0]],10);
  const state=lab.snapshot();assert.equal(state.history.length,10);assert.equal(state.history[0].status,'partial');
  assert.equal(state.history[0].result.modelError,'not installed');
});
test('a successful worker response without executed forecasts is not labeled complete',async()=>{
  const lab=controller(async id=>({...result(id),modelResult:{models:{short:{state:'insufficient_history'}}}}));
  await lab.start([scenarios[0]],10);assert.equal(lab.snapshot().history[0].status,'partial');
});

test('explicit replay capture failures remain partial even with executed forecasts',async()=>{
  for(const key of Object.keys(healthy())) {
    const fixture=result('baseline');fixture.run.health[key]=1;
    const lab=controller(async()=>fixture);await lab.start([scenarios[0]],160);
    const record=lab.snapshot().history[0];
    assert.equal(record.status,'partial',key);assert.equal(record.result,fixture);
    assert.deepEqual(record.qualityIssues,[`${key}: 1`]);
  }
});

test('missing or invalid capture quality is not treated as verified healthy',async()=>{
  for(const health of [undefined,{},null,{...healthy(),parseErrors:-1},{...healthy(),sourceErrors:'0'},{...healthy(),queueOverflows:NaN}]) {
    const fixture=result('baseline');fixture.run.health=health;
    const lab=controller(async()=>fixture);await lab.start([scenarios[0]],160);
    assert.equal(lab.snapshot().history[0].status,'partial');
    assert.match(lab.snapshot().history[0].qualityIssues[0],/unavailable or invalid/);
  }
});

test('normal filtering and model warmup counters do not imply capture failure',async()=>{
  const fixture=result('baseline');
  Object.assign(fixture.run.health,{unsupportedRecords:2,processedPackets:158,sourceRecords:160,reorderedPackets:1});
  const lab=controller(async()=>fixture);await lab.start([scenarios[0]],160);
  assert.equal(lab.snapshot().history[0].status,'complete');
  assert.deepEqual(lab.snapshot().history[0].qualityIssues,[]);
});
test('rule evidence comes from returned verdicts, not scenario labels',()=>{
  const no=evidence({protocol:'TCP',packets:2,initiatorAddress:'a',responderAddress:'b'});
  assert.equal(no.rule,'No deterministic rule match');
  const yes=evidence({attackVerdict:{severity:'high',summary:'Observed signature',detector:'rule-engine',evidence:['actual flag']}});
  assert.equal(yes.rule,'high · Observed signature');assert.match(yes.detail,/actual flag/);
});
test('lab IPC and renderer have no live-state mutations or global busy gate',()=>{
  const source=fs.readFileSync(path.join(__dirname,'../app.js'),'utf8');
  const bridge=source.slice(source.indexOf('window.ArgusSimulation ='),source.indexOf('\nasync function initialize()'));
  const lab=fs.readFileSync(path.join(__dirname,'../simulation-lab.mjs'),'utf8');
  assert.match(bridge,/run_simulation_scenario/);assert.match(bridge,/renderSimulationModelResult/);
  assert.doesNotMatch(bridge,/renderRun\(|state\.|setBusy\(/);
  assert.doesNotMatch(lab,/renderRun\(|state\.flows|state\.run|innerHTML|setInterval|fetch\(/);
  assert.doesNotMatch(source,/sim-continuous-toggle|simulationScenarioIds|triggerSimulation\(/);
});
test('mounted lab renders returned evidence and model results without touching a live dashboard',async()=>{
  class Element {
    constructor(tag){this.tag=tag;this.children=[];this.style={};this.attrs={};this.handlers={};this.value='160';}
    append(...items){this.children.push(...items);}
    replaceChildren(...items){this.children=[...items];}
    setAttribute(key,value){this.attrs[key]=value;}
    addEventListener(key,handler){this.handlers[key]=handler;}
    querySelectorAll(tag){return this.children.flatMap(child=>[...(child.tag===tag?[child]:[]),...child.querySelectorAll(tag)]);}
    setCustomValidity(){}reportValidity(){return true;}
  }
  const ids=['simulation-cards-container','sim-lab-status','sim-packet-count','sim-run-all-btn','sim-stop-btn','sim-reload-btn','simulation-status-card','sim-status-name','sim-stats-row','sim-flow-evidence','sim-model-results','sim-evidence-summary','sim-run-queue','sim-run-history'];
  const elements=Object.fromEntries(ids.map(id=>[id,new Element()]));
  const root={getElementById:id=>{assert.ok(elements[id],`unexpected access to ${id}`);return elements[id];},createElement:tag=>new Element(tag)};
  let modelsRendered=0;
  const fixture={...result('baseline'),packetCount:160,flowsCount:1,anomaliesObserved:0,highSeverityAnomalies:0,run:{health:{...healthy(),processedPackets:160,parseErrors:1},flows:[{protocol:'TCP',packets:160,initiatorAddress:'local',responderAddress:'fixture'}]}};
  const lab=mount(root,{list:async()=>scenarios,run:async()=>fixture,renderModels:value=>{assert.equal(value,fixture);modelsRendered++;}});
  await Promise.resolve();await Promise.resolve();
  assert.equal(elements['simulation-cards-container'].children.length,2);assert.equal(elements['sim-run-all-btn'].disabled,false);
  await lab.start([scenarios[0]],160);
  assert.equal(modelsRendered,1);assert.equal(elements['sim-flow-evidence'].children.length,1);
  assert.equal(elements['sim-stats-row'].children.length,5);assert.equal(elements['sim-stop-btn'].disabled,true);
  assert.match(elements['sim-lab-status'].textContent,/1 \/ 1 finished/);
  assert.match(elements['sim-evidence-summary'].textContent,/No live alerts/);
  assert.match(elements['sim-evidence-summary'].textContent,/Capture quality incomplete: parseErrors: 1/);
  assert.match(elements['sim-status-name'].textContent,/partial/);
  assert.match(elements['sim-lab-status'].textContent,/1 partial \(forecasts or capture quality\)/);
});
