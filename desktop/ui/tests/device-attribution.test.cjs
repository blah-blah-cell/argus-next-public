const {test} = require('node:test');
const assert = require('node:assert/strict');
const now = Date.parse('2026-11-05T12:00:00Z');
const iso = ms => new Date(ms).toISOString();
const socket = {protocol:'TCP',localAddress:'127.0.0.1',localPort:5000,remoteAddress:'1.2.3.4',remotePort:443,processId:42,processName:'browser',processStartedAt:iso(now-60000),processPath:'C:\\Apps\\browser.exe'};
const context = {networkStartedAt:iso(now-2000),networkCompletedAt:iso(now-1000),connections:[socket],detections:[]};
const flow = {uid:'a',protocol:'TCP',initiatorAddress:'127.0.0.1',initiatorPort:5000,responderAddress:'1.2.3.4',responderPort:443,firstTimestampNs:(now-3000)*1e6,lastTimestampNs:(now-1500)*1e6,capturedBytes:100};
const run = flows => ({live:true,flows});
test('matches both directions, deduplicates flow IDs and accounts unknown bytes separately', async () => {
  const {attributeFlows} = await import('../device-attribution.mjs');
  const reversed = {...flow,uid:'b',initiatorAddress:flow.responderAddress,initiatorPort:443,responderAddress:flow.initiatorAddress,responderPort:5000};
  const r = attributeFlows(context,run([flow,flow,reversed,{...flow,uid:'c',initiatorPort:999}]),now);
  assert.equal(r.matchedBytes,200); assert.equal(r.unknownBytes,100); assert.equal(r.rows[0].flows,2);
});
test('never attributes imported/simulated data, stale contexts, UDP or ambiguous owners', async () => {
  const {attributeFlows} = await import('../device-attribution.mjs');
  assert.equal(attributeFlows(context,{live:false,flows:[flow]},now).state,'not-live');
  assert.equal(attributeFlows(context,run([flow]),now+20000).state,'stale');
  assert.equal(attributeFlows(context,run([{...flow,protocol:'UDP'}]),now).matchedBytes,0);
  const r = attributeFlows({...context,connections:[socket,{...socket,processId:43}]},run([flow]),now);
  assert.equal(r.ambiguousFlows,1); assert.equal(r.matchedBytes,0);
});
test('rejects PID reuse, missing identity, future packets and long-lived flow totals', async () => {
  const {attributeFlows} = await import('../device-attribution.mjs');
  for (const c of [{...socket,processStartedAt:iso(now)}, {...socket,processStartedAt:null}, {...socket,processName:null}]) assert.equal(attributeFlows({...context,connections:[c]},run([flow]),now).matchedBytes,0);
  for (const f of [{...flow,firstTimestampNs:(now-60000)*1e6},{...flow,lastTimestampNs:(now+5000)*1e6}]) assert.equal(attributeFlows(context,run([f]),now).matchedBytes,0);
});
test('Defender linkage requires an exact full executable path, never a basename', async () => {
  const {relatedDetections} = await import('../device-attribution.mjs');
  const hits = [{processName:'browser.exe'}, {processName:'C:/Apps/BROWSER.exe'}, {processName:'C:\\Other\\browser.exe'}];
  assert.equal(relatedDetections(socket,hits).length,1);
  assert.equal(relatedDetections({...socket,processPath:null},hits).length,0);
});
