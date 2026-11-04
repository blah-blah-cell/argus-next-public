// Data and camera math only. No generated traffic, DOM or network access.
import {journeyPose} from './journey-pose.mjs?v=room-orbit-1';
export const LIMITS = Object.freeze({ nodes: 36, flows: 120, renderPixels: 1_300_000 });

export function hashId(value) {
  let hash = 2166136261;
  for (const char of String(value)) hash = Math.imul(hash ^ char.charCodeAt(0), 16777619);
  return hash >>> 0;
}

export function endpointPosition(address) {
  const a = hashId(address) / 4294967296;
  const b = hashId(`${address}/height`) / 4294967296;
  const radius = 3.5 + (hashId(`${address}/radius`) % 100) / 100;
  return [Math.cos(a * Math.PI * 2) * radius, (b - .5) * 3.4, Math.sin(a * Math.PI * 2) * radius];
}

export function observedGraph(input, protocol = 'all') {
  const usable = (Array.isArray(input) ? input : []).filter(flow =>
    typeof flow?.initiatorAddress === 'string' && flow.initiatorAddress.length > 0 &&
    typeof flow?.responderAddress === 'string' && flow.responderAddress.length > 0 &&
    Number.isFinite(flow.originalBytes) && flow.originalBytes >= 0 &&
    Number.isFinite(flow.packets) && flow.packets >= 0 &&
    (protocol === 'all' || String(flow.protocol).toLowerCase() === protocol));
  const ranked = [...usable].sort((a, b) => b.originalBytes - a.originalBytes || String(a.id).localeCompare(String(b.id)));
  const allEndpoints = new Set(usable.flatMap(flow => [flow.initiatorAddress, flow.responderAddress]));
  const nodes = new Map(), edges = new Map();
  let shownFlows = 0;
  for (const flow of ranked.slice(0, LIMITS.flows)) {
    const addresses = [...new Set([flow.initiatorAddress, flow.responderAddress])];
    if (nodes.size + addresses.filter(address => !nodes.has(address)).length > LIMITS.nodes) continue;
    for (const address of addresses) {
      if (!nodes.has(address)) nodes.set(address, { address, packets: 0, bytes: 0, flows: 0, position: endpointPosition(address) });
      const node = nodes.get(address);
      node.packets += flow.packets;
      node.bytes += flow.originalBytes;
      node.flows += 1;
    }
    const key = JSON.stringify([flow.initiatorAddress, flow.responderAddress]);
    if (!edges.has(key)) edges.set(key, { source: flow.initiatorAddress, target: flow.responderAddress, bytes: 0, flows: 0 });
    edges.get(key).bytes += flow.originalBytes;
    edges.get(key).flows += 1;
    shownFlows += 1;
  }
  return { nodes: [...nodes.values()], edges: [...edges.values()], totalFlows: usable.length,
    totalNodes: allEndpoints.size, shownFlows, omittedFlows: usable.length - shownFlows };
}

export function clampCamera(camera) {
  const finite = (value, fallback) => Number.isFinite(value) ? value : fallback;
  return { yaw: finite(camera.yaw, .3), pitch: Math.max(-1.2, Math.min(1.2, finite(camera.pitch, .15))),
    distance: Math.max(6, Math.min(21, finite(camera.distance, 11))) };
}

export function pixelRatio(width, height, deviceRatio = 1) {
  if (!(width > 0 && height > 0)) return 1;
  return Math.min(Math.max(.1, deviceRatio || 1), 2, Math.sqrt(LIMITS.renderPixels / (width * height)));
}

export function endpointSummary(graph, address) {
  return graph.nodes.find(node => node.address === address) ?? null;
}

export function storyPose(progress) {
  return journeyPose(progress);
}
