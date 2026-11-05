export function packetCount(value) {
  const count=Number(value);
  if(!Number.isInteger(count)||count<10||count>500)throw new Error('Choose a whole packet count from 10 to 500.');
  return count;
}
export function scenarioCatalog(value) {
  if(!Array.isArray(value))throw new Error('The runtime returned an invalid scenario catalog.');
  const ids=new Set();
  return value.map(item=>{
    if(!item||typeof item.id!=='string'||!item.id||ids.has(item.id)||typeof item.title!=='string')throw new Error('The scenario catalog contains an invalid or duplicate entry.');
    ids.add(item.id);return item;
  });
}
export function replayQualityIssues(health) {
  // These are explicit capture failures, not normal filtering or model warmup.
  const counters=['malformedRecords','reportedLostPackets','truncatedPackets','queueOverflows','parseErrors','sourceErrors'];
  if(!health||counters.some(key=>!Number.isSafeInteger(health[key])||health[key]<0))return ['Capture quality counters unavailable or invalid.'];
  return counters.filter(key=>health[key]>0).map(key=>`${key}: ${health[key]}`);
}
export function createLabController(execute,onChange=()=>{},now=()=>Date.now()) {
  let busy=false,stopping=false,queue=[],history=[],sequence=0;
  const snapshot=()=>({busy,stopping,queue:queue.map(item=>({...item})),history:[...history]});
  const emit=()=>onChange(snapshot());
  return {
    snapshot,
    stop(){if(busy){stopping=true;emit();}},
    async start(scenarios,value) {
      if(busy)return false;
      const count=packetCount(value),catalog=scenarioCatalog(scenarios);
      if(!catalog.length)throw new Error('No scenarios are available. Reload the catalog.');
      busy=true;stopping=false;queue=catalog.map(sc=>({id:sc.id,title:sc.title,status:'queued'}));emit();
      try {
        for(const item of queue) {
          if(stopping)break;
          item.status='running';emit();
          const start=now();let record;
          try {
            const result=await execute(item.id,count);
            if(result?.scenarioId!==item.id||!result.run||!Array.isArray(result.run.flows))throw new Error('Runtime returned an invalid or mismatched replay result.');
            const forecasts=Object.values(result.modelResult?.models||{}).filter(output=>output.state==='forecast_only');
            const qualityIssues=replayQualityIssues(result.run.health);
            const status=result.modelError||!forecasts.length||qualityIssues.length?'partial':'complete';
            record={key:++sequence,id:item.id,title:item.title,status,result,qualityIssues,elapsedMs:Math.max(0,now()-start),count};
          } catch(error) {
            record={key:++sequence,id:item.id,title:item.title,status:'failed',error:String(error),elapsedMs:Math.max(0,now()-start),count};
          }
          item.status=record.status;history=[record,...history].slice(0,10);emit();
        }
      } finally {
        queue.forEach(item=>{if(item.status==='queued')item.status='skipped';});busy=false;emit();
      }
      return true;
    },
  };
}
export function flowEvidence(flow) {
  const verdict=flow.attackVerdict;
  return {
    endpoints:`${flow.initiatorAddress??'Unknown'}:${flow.initiatorPort??'—'} → ${flow.responderAddress??'Unknown'}:${flow.responderPort??'—'}`,
    protocol:String(flow.protocol??'Unknown'),packets:Number.isFinite(flow.packets)?String(flow.packets):'Unavailable',
    rule:verdict?`${verdict.severity??'Unspecified'} · ${verdict.summary??verdict.category??'Rule match'}`:'No deterministic rule match',
    detail:verdict?[verdict.detector,verdict.detail,...(Array.isArray(verdict.evidence)?verdict.evidence:[])].filter(Boolean).join(' · '):'Forecasts are separate from rule verdicts.',
  };
}
export function mountSimulationLab(root,bridge) {
  const $=id=>root.getElementById(id),cards=$('simulation-cards-container');
  if(!cards)return;
  let catalog=[],selectedKey=null,lastKey=null,loading=false;
  const status=$('sim-lab-status'),count=$('sim-packet-count'),all=$('sim-run-all-btn'),stop=$('sim-stop-btn'),reload=$('sim-reload-btn');
  const make=(tag,text,className)=>{const el=root.createElement(tag);if(text!==undefined)el.textContent=text;if(className)el.className=className;return el;};
  function showRecord(record) {
    const panel=$('simulation-status-card');panel.style.display='block';
    $('sim-status-name').textContent=`${record.title} · ${record.status} · ${(record.elapsedMs/1000).toFixed(2)} s`;
    const stats=$('sim-stats-row'),evidence=$('sim-flow-evidence');stats.replaceChildren();evidence.replaceChildren();
    if(!record.result) {
      $('sim-model-results').textContent=record.error;
      $('sim-evidence-summary').textContent='Replay failed. No successful result is implied; earlier session runs remain available.';return;
    }
    const result=record.result;
    const outputs=Object.values(result.modelResult?.models||{});
    for(const [label,value] of [['Generated packets',result.packetCount],['Parsed packets',result.run.health?.processedPackets??'Unavailable'],['Flows',result.flowsCount],['Rule-matched flows',`${result.anomaliesObserved} · ${result.highSeverityAnomalies} critical`],['Forecast models',`${outputs.filter(output=>output.state==='forecast_only').length} / ${outputs.length}`]]) {
      const box=make('div',undefined,'sim-stat-box');box.append(make('span',label),make('strong',String(value)));stats.append(box);
    }
    bridge.renderModels(result);
    const flows=result.run.flows;
    const quality=record.qualityIssues?.length?` Capture quality incomplete: ${record.qualityIssues.join('; ')}.`:'';
    $('sim-evidence-summary').textContent=`${Math.min(flows.length,100)} of ${flows.length} replay flows shown.${quality} Temporary PCAP removed by the runtime after replay. No live alerts or model memory were updated.`;
    for(const flow of flows.slice(0,100)) {
      const row=make('tr'),item=flowEvidence(flow);
      row.append(make('td',item.endpoints),make('td',item.protocol),make('td',item.packets));
      const cell=make('td'),details=make('details');details.append(make('summary',item.rule),make('p',item.detail));cell.append(details);row.append(cell);evidence.append(row);
    }
  }
  function paint(state) {
    count.disabled=state.busy;all.disabled=state.busy||!catalog.length||loading;reload.disabled=state.busy||loading;stop.disabled=!state.busy||state.stopping;
    cards.querySelectorAll('button').forEach(button=>button.disabled=state.busy||loading);
    cards.setAttribute('aria-busy',String(loading));
    const done=state.queue.filter(item=>['complete','partial','failed'].includes(item.status)).length;
    if(state.busy)status.textContent=state.stopping?'Stopping after the current replay finishes. No new scenario will start.':`Running ${state.queue.find(item=>item.status==='running')?.title??'suite'} · ${done} / ${state.queue.length} finished`;
    else if(state.queue.length)status.textContent=`${done} / ${state.queue.length} finished · ${state.queue.filter(item=>item.status==='failed').length} failed · ${state.queue.filter(item=>item.status==='partial').length} partial (forecasts or capture quality) · ${state.queue.filter(item=>item.status==='skipped').length} skipped`;
    const queue=$('sim-run-queue');queue.replaceChildren();
    for(const item of state.queue)queue.append(make('li',`${item.title} — ${item.status}`));
    const latest=state.history[0];
    if(latest&&latest.key!==lastKey){lastKey=latest.key;selectedKey=latest.key;showRecord(latest);}
    const history=$('sim-run-history');history.replaceChildren();
    for(const record of state.history) {
      const li=make('li'),button=make('button',`${record.title} · ${record.status} · ${record.count} packets`,'button secondary');
      button.type='button';button.setAttribute('aria-pressed',String(record.key===selectedKey));
      button.addEventListener('click',()=>{selectedKey=record.key;showRecord(record);history.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));});li.append(button);history.append(li);
    }
  }
  const controller=createLabController((id,n)=>bridge.run(id,n),paint);
  const run=scenarios=>{try {count.setCustomValidity('');if(!count.reportValidity())return;void controller.start(scenarios,count.value).catch(error=>{status.textContent=String(error);});}catch(error){status.textContent=String(error);}};
  count.addEventListener('input',()=>count.setCustomValidity(''));
  stop.addEventListener('click',()=>controller.stop());all.addEventListener('click',()=>run(catalog));
  async function load() {
    if(loading||controller.snapshot().busy)return;
    loading=true;status.textContent='Loading registered local scenarios…';paint(controller.snapshot());
    try {
      catalog=scenarioCatalog(await bridge.list());cards.replaceChildren();
      for(const sc of catalog) {
        const card=make('article',undefined,'simulation-card'),header=make('div',undefined,'sim-card-header');
        header.append(make('h3',sc.title),make('span',sc.category??'Local fixture','sim-tactic-pill'));
        const button=make('button','Run scenario','button secondary');button.type='button';button.setAttribute('aria-label',`Run ${sc.title}`);button.addEventListener('click',()=>run([sc]));
        card.append(header,make('p',sc.description??'','sim-description'),make('p',`${sc.technique?.techniqueId??''} · ${sc.technique?.techniqueName??'Local replay'}`,'sim-technique-row'),button);cards.append(card);
      }
      status.textContent=catalog.length?`${catalog.length} scenarios ready, including the baseline when registered. Choose a scenario or run the full suite.`:'No scenarios registered. Reload to retry.';
    } catch(error) {catalog=[];cards.replaceChildren(make('p',`Scenarios unavailable: ${error}. Use Reload scenarios to retry.`));status.textContent='Local runtime unavailable. No simulation has run.';}
    finally{loading=false;paint(controller.snapshot());}
  }
  reload.addEventListener('click',()=>void load());void load();return controller;
}
if(typeof document!=='undefined')mountSimulationLab(document,window.ArgusSimulation);
