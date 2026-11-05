const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'../app.js'),'utf8');
const html=fs.readFileSync(path.join(__dirname,'../index.html'),'utf8');
function extract(name){
  const start=source.search(new RegExp(`(?:async )?function ${name}\\(`));
  assert.ok(start>=0,name);
  const rest=source.slice(start);
  const end=rest.slice(1).search(/\n(?:async )?function /);
  return end<0?rest:rest.slice(0,end+1);
}
function run(name='file.pcap',id=1){return {source:name,session:name,detector:{id:'rules',trained:false},flows:[{id,protocol:'TCP',initiatorAddress:'10.0.0.1',responderAddress:'10.0.0.2'}],health:Object.fromEntries(['sourceRecords','normalizedPackets','processedPackets','flows','nonIpRecords','unsupportedRecords','malformedRecords','reportedLostPackets','truncatedPackets','reorderedPackets','queueOverflows','parseErrors','sourceErrors'].map(k=>[k,0]))};}
function harness(invoke){
  const elements=new Map();
  const state={activeCapturePath:'old.pcap',run:run('old.pcap'),flowPagination:{page:3},lastModelResult:{old:true}};
  const renders=[];
  const context=vm.createContext({state,invoke,$(id){if(!elements.has(id))elements.set(id,{textContent:'',disabled:false,setAttribute(){},replaceChildren(){}});return elements.get(id);},notice(){},closeFlow(){},renderModelActivity(){},renderRun(value,simulation,live,offline){state.run=value;state.runIsOfflineReplay=offline;renders.push(value);}});
  for(const name of ['updateImportControl','clearReplayModelOutput','validateReplayRun','importReplayCapture'])vm.runInContext(extract(name),context);
  return {state,context,elements,renders};
}
test('cancel preserves source, run and model results without replay',async()=>{
  const calls=[];const h=harness(async command=>{calls.push(command);return null;});
  const previous=h.state.run;
  assert.equal(await h.context.importReplayCapture(),false);
  assert.deepEqual(calls,['choose_replay_capture']);
  assert.equal(h.state.run,previous);assert.equal(h.state.activeCapturePath,'old.pcap');
  assert.ok(h.state.lastModelResult);assert.equal(h.state.importBusy,false);
  assert.match(h.elements.get('import-replay-status').textContent,/cancelled/);
});
test('picker and replay failures preserve displayed source and evidence',async()=>{
  for(const failedCommand of ['choose_replay_capture','replay_capture']){
    const h=harness(async command=>{if(command===failedCommand)throw Error('denied');return 'new.pcap';});
    const previous=h.state.run;
    assert.equal(await h.context.importReplayCapture(),false);
    assert.equal(h.state.run,previous);assert.equal(h.state.activeCapturePath,'old.pcap');assert.ok(h.state.lastModelResult);
    assert.equal(h.state.importBusy,false);assert.match(h.elements.get('import-replay-status').textContent,/Import failed:.*denied/);
  }
});
test('repeated imports replace evidence, clear old model output and do not call live or learning IPC',async()=>{
  const calls=[];let n=0;
  const h=harness(async(command,args)=>{calls.push([command,args]);return command==='choose_replay_capture'?`file-${++n}.pcap`:run(args.path,n);});
  assert.equal(await h.context.importReplayCapture(),true);
  assert.equal(h.state.lastModelResult,null);assert.equal(h.state.flowPagination.page,1);
  assert.equal(await h.context.importReplayCapture(),true);
  assert.equal(h.state.run.source,'file-2.pcap');assert.equal(h.state.activeCapturePath,'file-2.pcap');
  assert.equal(h.state.run.flows.length,1);assert.equal(h.state.runIsOfflineReplay,true);
  assert.deepEqual(calls.map(c=>c[0]),['choose_replay_capture','replay_capture','choose_replay_capture','replay_capture']);
});
test('invalid runtime evidence is rejected before committing source or clearing model results',async()=>{
  const h=harness(async command=>command==='choose_replay_capture'?'broken.pcap':{source:'bad',flows:[]});
  const previous=h.state.run;
  assert.equal(await h.context.importReplayCapture(),false);assert.equal(h.state.run,previous);
  assert.equal(h.state.activeCapturePath,'old.pcap');assert.ok(h.state.lastModelResult);
});
test('a mismatched runtime source cannot bind old evidence to a newly selected file',async()=>{
  const h=harness(async command=>command==='choose_replay_capture'?'selected.pcap':run('different.pcap'));
  const previous=h.state.run;
  assert.equal(await h.context.importReplayCapture(),false);
  assert.equal(h.state.run,previous);assert.equal(h.state.activeCapturePath,'old.pcap');assert.ok(h.state.lastModelResult);
  assert.match(h.elements.get('import-replay-status').textContent,/does not match/);
});
test('monitoring, stopping, initialization and other busy operations reject import before IPC',async()=>{
  for(const flag of ['initializing','rolling','monitorStopping','busy','rollingTickBusy','monitorControlBusy','onboardingSaving']){
    let calls=0;const h=harness(async()=>{calls++;});h.state[flag]=true;
    assert.equal(await h.context.importReplayCapture(),false,flag);assert.equal(calls,0,flag);
  }
});
test('concurrent imports serialize and rolling commands/ticks cannot run during the file picker',async()=>{
  let resolvePicker;let calls=0;const h=harness(async command=>{calls++;if(command==='choose_replay_capture')return new Promise(resolve=>{resolvePicker=resolve;});return run();});
  const first=h.context.importReplayCapture();
  assert.equal(h.state.importBusy,true);assert.equal(h.elements.get('rolling-button').disabled,true);
  assert.equal(await h.context.importReplayCapture(),false);assert.equal(calls,1);
  for(const name of ['toggleRollingCapture','ensureRollingCapture','rollingTick']){
    vm.runInContext(extract(name),h.context);await h.context[name]();
  }
  assert.equal(calls,1);resolvePicker(null);await first;
  assert.equal(h.elements.get('rolling-button').disabled,false);
});
test('offline source and model caveats are visible with keyboard-accessible controls and persistent status',()=>{
  assert.match(html,/<button[^>]+id="import-replay-button"[^>]+type="button"[^>]+aria-describedby="import-replay-hint import-replay-status"/);
  assert.match(html,/id="import-replay-status" role="status" aria-live="polite"/);
  assert.match(extract('renderRun'),/OFFLINE REPLAY/);
  assert.match(extract('renderEvidence'),/Not run for this file/);
  assert.match(extract('renderAlerts'),/state\.runIsOfflineReplay \? null/);
  assert.match(extract('renderTable'),/Inspect flow.*rule and capture evidence/);
  assert.match(extract('openFlow'),/attackVerdict\.evidence\.join/);
  const workspace=fs.readFileSync(path.join(__dirname,'../spatial/workspace.mjs'),'utf8');
  const overview=workspace.match(/const overview = new Set\(\[([^\]]+)\]\)/)[1];
  assert.match(overview,/'offline-replay'/,'import remains visible in the loaded overview workspace');
  const header=html.slice(html.indexOf('<header id="overview">'),html.indexOf('</header>'));
  assert.match(header,/id="analysis-source"/,'source remains visible when the flow and evidence sections replace overview');
});

function node(){return {children:[],textContent:'',value:'',style:{},classList:{add(){},remove(){}},setAttribute(){},addEventListener(){},replaceChildren(){this.children=[];},append(...children){this.children.push(...children);},appendChild(child){this.children.push(child);}};}
test('offline malicious flow is not masked by a current live allowlist and drawer handlers cannot mutate policy',async()=>{
  const elements=new Map();const get=id=>{if(!elements.has(id))elements.set(id,node());return elements.get(id);};
  let calls=0;
  const state={runIsOfflineReplay:true,activeCapturePath:'attack.pcap',flowPagination:{page:1},blacklistRules:[{active:true,target:'10.0.0.1',action:'allow'}]};
  const flow={...run().flows[0],attackVerdict:{summary:'SYN burst rule',severity:'critical',evidence:['100 SYNs']},packets:100,initiatorPackets:100,responderPackets:0};
  const context=vm.createContext({state,$:get,invoke:async()=>{calls++;},currentFlowPayload:null,activeCaptureSupportsFlow:()=>false,matchFlowFilter:()=>true,sortFlows:v=>v,paginateFlows:v=>({items:v,page:1,pageSize:25,totalPages:1,totalItems:v.length}),renderFlowSortState(){},renderPaginationControls(){},endpoint:(a,p)=>`${a}:${p}`,formatNumber:String,formatBytes:String,formatDuration:String,document:{createElement:node,createDocumentFragment:node,querySelector:()=>null}});
  vm.runInContext(extract('renderTable'),context);vm.runInContext(extract('openFlow'),context);
  context.renderTable([flow]);
  const row=get('flow-rows').children[0].children[0];
  assert.equal(row.children[7].textContent,'SYN burst rule');
  context.openFlow(flow);
  assert.match(get('quality-evidence').textContent,/100 SYNs/);
  assert.match(get('drawer-block-status').textContent,/Offline replay is read-only/);
  assert.equal(get('drawer-toggle-block-btn').disabled,true);assert.equal(get('drawer-allow-btn').disabled,true);
  await get('drawer-toggle-block-btn').onclick();await get('drawer-allow-btn').onclick();
  assert.equal(calls,0);
});
