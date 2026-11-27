// Conservative, ephemeral correlation. Never treats a socket snapshot as packet ownership proof.
export const MAX_CONTEXT_AGE_MS = 10_000;
function ip(value) {
  const s = String(value || '').toLowerCase();
  if (!s.includes(':')) return s;
  try { return new URL(`http://[${s}]/`).hostname; } catch { return s; }
}
function endpoint(a, p, b, q) { return ip(a) === ip(b) && Number(p) === Number(q); }
function path(value) {
  const s = String(value || '').trim().replaceAll('/', '\\').toLowerCase();
  return /^[a-z]:\\/.test(s) ? s : null;
}
export function relatedDetections(connection, detections) {
  const executable = path(connection.processPath);
  return executable ? detections.filter(d => path(d.processName) === executable) : [];
}
export function attributeFlows(context, run, now = Date.now()) {
  const empty = {state: 'unavailable', rows: [], matchedBytes: 0, unknownBytes: 0, unknownFlows: 0, ambiguousFlows: 0};
  if (!run?.live) return {...empty, state: 'not-live'};
  const start = Date.parse(context?.networkStartedAt), end = Date.parse(context?.networkCompletedAt);
  if (!Number.isFinite(start) || !Number.isFinite(end) || start > end || end > now || now - start > MAX_CONTEXT_AGE_MS) return {...empty, state: 'stale'};
  const result = {...empty, state: 'ready'};
  const groups = new Map(), seen = new Set();
  for (const flow of (run.flows || []).slice(0, 5000)) {
    if (!flow.uid || seen.has(flow.uid)) continue;
    seen.add(flow.uid);
    const bytes = Number.isSafeInteger(flow.capturedBytes) && flow.capturedBytes >= 0 ? flow.capturedBytes : 0;
    const first = Number(flow.firstTimestampNs) / 1e6, last = Number(flow.lastTimestampNs) / 1e6;
    // Only whole short flow intervals close to the socket observation are eligible.
    const timely = Number.isFinite(first) && Number.isFinite(last) && first <= last && first >= start - MAX_CONTEXT_AGE_MS && last <= end + MAX_CONTEXT_AGE_MS && last <= now;
    const candidates = timely && String(flow.protocol).toUpperCase() === 'TCP' ? (context.connections || []).filter(c => c.protocol === 'TCP' && c.processId > 0 && c.processName && Date.parse(c.processStartedAt) <= first && (
      (endpoint(flow.initiatorAddress, flow.initiatorPort, c.localAddress, c.localPort) && endpoint(flow.responderAddress, flow.responderPort, c.remoteAddress, c.remotePort)) ||
      (endpoint(flow.responderAddress, flow.responderPort, c.localAddress, c.localPort) && endpoint(flow.initiatorAddress, flow.initiatorPort, c.remoteAddress, c.remotePort))
    )) : [];
    const owners = new Map(candidates.map(c => [`${c.processId}/${c.processStartedAt}`, c]));
    if (owners.size !== 1) {
      result.unknownFlows++; result.unknownBytes += bytes;
      if (owners.size > 1) result.ambiguousFlows++;
      continue;
    }
    const [key, c] = owners.entries().next().value;
    if (!groups.has(key)) groups.set(key, {name: c.processName, pid: c.processId, bytes: 0, flows: 0, destinations: new Set(), last: 0, detections: relatedDetections(c, context.detections || []).length});
    const g = groups.get(key);
    g.bytes += bytes; g.flows++; g.last = Math.max(g.last, last);
    g.destinations.add(`[${c.remoteAddress}]:${c.remotePort}`);
    result.matchedBytes += bytes;
  }
  result.rows = [...groups.values()].map(g => ({...g, destinations: [...g.destinations]})).sort((a, b) => b.bytes - a.bytes);
  return result;
}
