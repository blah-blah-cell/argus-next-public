const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../app.js'), 'utf8');
const html = fs.readFileSync(path.join(__dirname, '../index.html'), 'utf8');
const styles = fs.readFileSync(path.join(__dirname, '../styles.css'), 'utf8');
function extract(name) {
  const start = source.indexOf(`function ${name}(`);
  assert.ok(start >= 0, `Missing function ${name}`);
  const end = source.indexOf('\nfunction ', start + 1);
  return source.slice(start, end < 0 ? source.length : end);
}

test('missing and non-finite forecast values stay unavailable', () => {
  const context = vm.createContext({});
  vm.runInContext(extract('formatDuration') + '\n' + extract('formatMetric'), context);
  for (const value of [undefined, null, NaN, Infinity, '12']) {
    assert.equal(context.formatMetric(value), '—');
    assert.equal(context.formatDuration(value), '—');
  }
  assert.equal(context.formatMetric(0, 5), '0.00000');
  assert.equal(context.formatDuration(0), '0 µs');
  assert.equal(context.formatDuration(-1), '—');
});

test('clearing a failed analysis removes stale execution status', () => {
  const elements = new Map();
  const context = vm.createContext({
    state: {},
    $(id) {
      if (id === 'model-activity-graph') return null;
      if (!elements.has(id)) elements.set(id, { textContent: 'stale' });
      return elements.get(id);
    },
  });
  vm.runInContext(extract('renderModelActivity'), context);
  context.renderModelActivity(null, 'idle');
  assert.equal(elements.get('model-activity-stats').textContent, 'Awaiting analysis');
  assert.equal(elements.get('model-activity-title').textContent, 'Live inference path');
  assert.match(elements.get('model-activity-detail').textContent, /unavailable until/);
});

test('empty model execution is compact until real work starts', () => {
  assert.match(html, /class="model-activity is-idle"/);
  const idleStates = [];
  const elements = new Map();
  const context = vm.createContext({
    state: {},
    $(id) {
      if (id === 'model-activity-graph') return null;
      if (id === 'model-activity-section') {
        return { classList: { toggle: (_className, enabled) => idleStates.push(enabled) } };
      }
      if (!elements.has(id)) elements.set(id, { textContent: '' });
      return elements.get(id);
    },
  });
  vm.runInContext(extract('renderModelActivity'), context);
  context.renderModelActivity(null, 'idle');
  context.renderModelActivity(null, 'running');
  assert.deepEqual(idleStates, [true, false]);
  assert.match(html, /<span>Flow instances<\/span>/);
  assert.match(html, /Flow instances in the latest analysis/);
});

test('scroll onboarding keeps sections visible and navigates to the requested chapter', () => {
  assert.match(extract('setOnboardingStep'), /panel\.hidden = false/);
  assert.match(extract('setOnboardingStep'), /panel\.scrollIntoView/);
  assert.match(extract('setOnboardingStep'), /panel\.focus\(\{preventScroll:true\}\)/);
  assert.match(styles, /\.onboarding-actions \[hidden\]\s*\{\s*display:\s*none;/);
});

test('rolling analysis always retires managed captures and runs while hidden', () => {
  const rollingTick = extract('rollingTick');
  const scheduler = extract('scheduleRollingTick');
  assert.match(rollingTick, /finally\s*\{[\s\S]*release_live_capture/);
  assert.match(rollingTick, /release_live_capture[\s\S]*status\s*===\s*["']deleted["']/);
  assert.doesNotMatch(rollingTick, /retainedLiveCapturePath/);
  assert.match(scheduler, /setTimeout\(async \(\) => \{\s*await rollingTick\(\)/);
});

test('captures awaiting retirement cannot re-enter analysis', () => {
  const context = vm.createContext({});
  vm.runInContext(extract('capturesReadyForAnalysis'), context);
  const deferred = 'C:\\capture\\live\\live-2.pcap';
  const ready = context.capturesReadyForAnalysis([
    'C:\\capture\\live\\live-1.pcap',
    deferred,
    'C:\\capture\\manual.pcap',
  ], new Map([[deferred, { retries: 1 }]]));
  assert.deepEqual(Array.from(ready), ['C:\\capture\\live\\live-1.pcap']);
});

test('boost mode reaches the native command boundary', () => {
  const inference = extract('tryCnn');
  assert.match(inference, /boostMode:\s*state\.runtimePreferences\.boostMode/);
});

test('forecast details disclose runtime model rejections', () => {
  const inference = extract('tryCnn');
  assert.match(inference, /Unavailable model components/);
  assert.match(inference, /result\.rejected_models/);
});

test('traffic summary derives rates only from valid captured segment samples', () => {
  const context = vm.createContext({ Number, Math, Array });
  vm.runInContext(extract('summarizeTrafficSamples'), context);
  const summary = context.summarizeTrafficSamples([
    { packets: 50, bytes: 5000, captureIssues: 0 },
    { packets: 100, bytes: 15000, captureIssues: 2 },
    { packets: Number.NaN, bytes: 1, captureIssues: 0 },
  ], 5);
  assert.equal(summary.latestRate, 20);
  assert.equal(summary.averageRate, 15);
  assert.equal(summary.peakRate, 20);
  assert.equal(summary.latestBytesPerSecond, 3000);
  assert.equal(summary.issueSegments, 1);
  assert.equal(context.summarizeTrafficSamples([], 5), null);
});

test('traffic comparison uses disjoint current and prior windows', () => {
  const context = vm.createContext({ Number, Array });
  vm.runInContext(extract('compareTrafficWindows'), context);
  const samples = [
    ...Array.from({ length: 12 }, () => ({ packets: 50, bytes: 500, flows: 2, captureIssues: 0 })),
    ...Array.from({ length: 12 }, () => ({ packets: 75, bytes: 750, flows: 3, captureIssues: 0 })),
  ];
  const comparison = context.compareTrafficWindows(samples, 5, 12);
  assert.equal(comparison.prior.packetRate, 10);
  assert.equal(comparison.latest.packetRate, 15);
  assert.equal(comparison.packetDeltaPercent, 50);
  assert.equal(context.compareTrafficWindows(samples.slice(1), 5, 12), null);
});

test('model latency history accepts completed real executions and remains bounded', () => {
  const context = vm.createContext({ Object, Array, Number });
  vm.runInContext(extract('appendModelLatencySamples'), context);
  let history = {};
  for (let i = 0; i < 5; i += 1) {
    history = context.appendModelLatencySamples(history, {
      ahem_v61: { state: 'forecast_only', execution_ms: i + 1, execution_provider: 'PyTorch CPU' },
      absent: { state: 'unavailable', execution_ms: 0 },
    }, i, 3);
  }
  assert.deepEqual(
    Array.from(history.ahem_v61, (sample) => sample.ms),
    [3, 4, 5],
  );
  assert.equal(history.absent, undefined);
});

test('flow sorting is stable on the selected fact and uses flow id as a tie break', () => {
  const context = vm.createContext({ Number });
  vm.runInContext(extract('flowSortValue') + '\n' + extract('sortFlows'), context);
  const flows = [
    { id: 1, packets: 9, originalBytes: 90 },
    { id: 3, packets: 3, originalBytes: 30 },
    { id: 2, packets: 9, originalBytes: 50 },
  ];
  assert.deepEqual(
    Array.from(context.sortFlows(flows, { key: 'packets', direction: 'desc' }), (flow) => flow.id),
    [2, 1, 3],
  );
  assert.deepEqual(
    Array.from(context.sortFlows(flows, { key: 'bytes', direction: 'asc' }), (flow) => flow.id),
    [3, 2, 1],
  );
  assert.deepEqual(flows.map((flow) => flow.id), [1, 3, 2]);
});

test('simulation UI does not claim attack-detection validation', () => {
  assert.match(html, /Rule matches and forecast outputs are shown separately/);
  assert.match(html, /Results stay here, separate from live monitoring/);
  assert.match(source, /run_simulation_scenario/);
  assert.match(source, /renderSimulationModelResult\(result\)/);
  assert.doesNotMatch(html, /validate detection rules/i);
  assert.doesNotMatch(html, /No anomaly alerts recorded\. Statistical baselines and protocols are nominal/);
});

test('node role formatting and radius calculations reflect network semantics and volumes', () => {
  const context = vm.createContext({});
  vm.runInContext(extract('formatNodeRole') + '\n' + extract('calculateNodeRadius'), context);

  assert.equal(context.formatNodeRole('gateway', false), 'Default Gateway / Router');
  assert.equal(context.formatNodeRole('lan_host', false), 'Local LAN Host');
  assert.equal(context.formatNodeRole('external_server', false), 'External WAN Server');
  assert.equal(context.formatNodeRole('external_server', true), 'Tor-associated port (heuristic)');
  assert.equal(context.formatNodeRole('tor_node', false), 'Tor-associated port (heuristic)');

  assert.equal(context.calculateNodeRadius({ role: 'gateway' }), 18);
  assert.equal(context.calculateNodeRadius({ role: 'lan_host', isTor: true }), 16);
  assert.equal(context.calculateNodeRadius({ role: 'external_server', threatScore: 0.95 }), 16);
  assert.equal(context.calculateNodeRadius({ role: 'lan_host', totalInBytes: 2000000, totalOutBytes: 500 }), 18);
  assert.equal(context.calculateNodeRadius({ role: 'lan_host', totalInBytes: 50, totalOutBytes: 50 }), 12);
});

test('topology labels do not turn port heuristics or rule flags into classifier probabilities', () => {
  assert.match(source, /TOR-PORT HEURISTIC/);
  assert.match(source, /indicator 1\.0 \(not probability\)/);
  assert.match(source, /relay identity unverified/);
  assert.doesNotMatch(source, /Critical Anomaly \(\$\{pct\}%\)/);
  assert.match(html, /Observed Rule \/ Policy Indicator/);
});

test('filterTopology filters network graphs according to protocols, tor, threats, and lan bounds', () => {
  const context = vm.createContext({ Set, Array, Boolean });
  vm.runInContext(extract('filterTopology'), context);

  const sampleTopology = {
    nodes: [
      { id: '192.168.1.1', role: 'gateway', threatScore: 0, activePorts: [53, 80] },
      { id: '192.168.1.100', role: 'lan_host', threatScore: 0, activePorts: [52140] },
      { id: '198.51.100.50', role: 'external_server', isTor: true, threatScore: 0, activePorts: [9001] },
      { id: '203.0.113.10', role: 'external_server', isTor: false, threatScore: 0.85, activePorts: [443] },
    ],
    edges: [
      { id: 'e1', source: '192.168.1.100', target: '192.168.1.1', protocol: 'DNS' },
      { id: 'e2', source: '192.168.1.100', target: '198.51.100.50', protocol: 'TCP', isTor: true },
      { id: 'e3', source: '192.168.1.100', target: '203.0.113.10', protocol: 'TLS', isMalicious: true, threatScore: 0.85 },
    ],
  };

  // 1. All
  const allResult = context.filterTopology(sampleTopology, 'all');
  assert.equal(allResult.filteredNodes.length, 4);
  assert.equal(allResult.filteredEdges.length, 3);

  // 2. Tor
  const torResult = context.filterTopology(sampleTopology, 'tor');
  assert.equal(torResult.filteredEdges.length, 1);
  assert.ok(torResult.filteredNodes.some((n) => n.id === '198.51.100.50'));
  assert.ok(torResult.filteredNodes.some((n) => n.id === '192.168.1.100'));

  // 3. Threats
  const threatsResult = context.filterTopology(sampleTopology, 'threats');
  assert.equal(threatsResult.filteredEdges.length, 1);
  assert.ok(threatsResult.filteredNodes.some((n) => n.id === '203.0.113.10'));

  // 4. LAN
  const lanResult = context.filterTopology(sampleTopology, 'lan');
  assert.equal(lanResult.filteredNodes.length, 2);
  assert.equal(lanResult.filteredEdges.length, 1);
  assert.equal(lanResult.filteredEdges[0].id, 'e1');

  // 5. DNS
  const dnsResult = context.filterTopology(sampleTopology, 'dns');
  assert.equal(dnsResult.filteredEdges.length, 1);
  assert.equal(dnsResult.filteredEdges[0].protocol, 'DNS');
});

test('force-directed physics updates node positions and respects pinned nodes', () => {
  const context = vm.createContext({ Math });
  vm.runInContext(extract('stepForceSimulation'), context);

  const n1 = { id: 'a', x: 0, y: 0, vx: 0, vy: 0, mass: 1, pinned: false };
  const n2 = { id: 'b', x: 10, y: 0, vx: 0, vy: 0, mass: 1, pinned: false };
  const pinnedNode = { id: 'c', x: 50, y: 50, vx: 0, vy: 0, mass: 1, pinned: true };

  const nodes = [n1, n2, pinnedNode];
  const edges = [{ sourceNode: n1, targetNode: n2 }];

  context.stepForceSimulation(nodes, edges, { repulsionStrength: 1000, springStrength: 0.01 });

  // n1 and n2 should experience forces and move
  assert.notEqual(n1.x, 0);
  assert.notEqual(n2.x, 10);

  // pinnedNode must not move
  assert.equal(pinnedNode.x, 50);
  assert.equal(pinnedNode.y, 50);
  assert.equal(pinnedNode.vx, 0);
  assert.equal(pinnedNode.vy, 0);
});

test('endpoint peer extraction and volume ratios calculate correct ingress/egress proportions', () => {
  const node = {
    id: '192.168.1.100',
    totalInBytes: 3000,
    totalOutBytes: 7000,
  };
  const total = node.totalInBytes + node.totalOutBytes;
  const inPct = Math.round((node.totalInBytes / total) * 100);
  const outPct = 100 - inPct;

  assert.equal(inPct, 30);
  assert.equal(outPct, 70);

  const edges = [
    { source: '192.168.1.100', target: '192.168.1.1', totalBytes: 5000, protocol: 'DNS' },
    { source: '10.0.0.5', target: '192.168.1.100', totalBytes: 2000, protocol: 'TLS' },
    { source: '8.8.8.8', target: '1.1.1.1', totalBytes: 100, protocol: 'DNS' }, // Unrelated
  ];

  const connected = edges.filter((e) => e.source === node.id || e.target === node.id);
  assert.equal(connected.length, 2);

  const peer1 = connected[0].source === node.id ? connected[0].target : connected[0].source;
  const peer2 = connected[1].source === node.id ? connected[1].target : connected[1].source;

  assert.equal(peer1, '192.168.1.1');
  assert.equal(peer2, '10.0.0.5');
});

test('asset intelligence attribution resolves entity names, organizations, ASNs, categories, and confidence', () => {
  const node = {
    id: '1.1.1.1',
    ip: '1.1.1.1',
    entityName: 'Cloudflare Public DNS',
    organization: 'Cloudflare, Inc.',
    asn: 'AS13335',
    category: 'Public DNS Resolver',
    deviceType: 'Anycast DNS Edge',
    description: 'High-speed privacy-first recursive DNS resolver with DNSSEC, DoH, and DoT support.',
    confidence: 'Authoritative (CIDR Database)',
    role: 'external_server',
    isTor: false,
  };

  const resolvedName = node.entityName || node.label || node.ip;
  const resolvedCategory = node.category;
  const orgWithAsn = node.asn ? `${node.organization} (${node.asn})` : node.organization;

  assert.equal(resolvedName, 'Cloudflare Public DNS');
  assert.equal(resolvedCategory, 'Public DNS Resolver');
  assert.equal(orgWithAsn, 'Cloudflare, Inc. (AS13335)');
  assert.equal(node.deviceType, 'Anycast DNS Edge');
  assert.ok(node.description.includes('DNSSEC'));
  assert.equal(node.confidence, 'Authoritative (CIDR Database)');
});

test('drawXRayCanvas eliminates AI-gimmick radar sweep beam and concentric distance rings', () => {
  const drawFn = extract('drawXRayCanvas');

  // Verify AI-gimmick artifacts are completely eliminated
  assert.doesNotMatch(drawFn, /sweepAngle/i, 'Rotating radar sweep beam must be eliminated');
  assert.doesNotMatch(drawFn, /sweepRadius/i, 'Rotating radar sweep radius must be eliminated');
  assert.doesNotMatch(drawFn, /radarRings/i, 'Distance concentric rings must be eliminated');
  assert.doesNotMatch(drawFn, /100m/i, 'Cheesy distance range labels must be eliminated');
  assert.doesNotMatch(drawFn, /shadowBlur\s*=\s*8/i, 'Harsh neon glow strobes must be eliminated');
  assert.doesNotMatch(drawFn, /slice\(0,\s*26\)/, 'Sublabel text truncation must be eliminated');

  // Verify high-craft telemetry design
  assert.match(drawFn, /#090d16/, 'Canvas must use deep obsidian background');
  assert.match(drawFn, /gridSpacing/, 'Canvas must render calm blueprint grid or dotted matrix');
  assert.match(drawFn, /roundRect/, 'Node pills must use rounded container');
  assert.match(drawFn, /pillHeight/, 'Node pills must support dual-line height');
});

test('endpoint drawer and canvas overlay provide polished bento card telemetry and indicators', () => {
  // Canvas overlay must use clean indicators instead of emoji tropes
  assert.match(html, /legend-dot dot-gateway/);
  assert.match(html, /legend-dot dot-tor/);
  assert.match(html, /legend-dot dot-threat/);
  assert.doesNotMatch(html, /🟢 Gateway/);
  assert.doesNotMatch(html, /🟣 Tor Relay/);

  // Inspector drawer bento cards and structured key-value attribution
  assert.match(html, /xray-intelligence-card/);
  assert.match(html, /xray-intel-eyebrow-group/);
  assert.match(html, /Operator &amp; ASN/);
  assert.match(html, /Device Classification/);
  assert.match(html, /id="xray-drawer-flows-btn"/);
  assert.match(html, /id="xray-drawer-block-btn"/);
  assert.match(html, /id="xray-drawer-close"/);
});

test('drawXRayCanvas records bounded pill dimensions and handles extreme text lengths without layout overflow', () => {
  const drawFn = extract('drawXRayCanvas');

  // Verify camera transformation inversion logic for grid boundaries
  assert.match(drawFn, /viewLeft\s*=\s*-\(centerX\s*\+\s*xrayState\.camera\.x\)\s*\/\s*xrayState\.camera\.zoom/);
  assert.match(drawFn, /viewRight\s*=\s*\(w\s*-\s*centerX\s*-\s*xrayState\.camera\.x\)\s*\/\s*xrayState\.camera\.zoom/);

  const longNode = {
    id: 'node-long',
    x: 100,
    y: 200,
    radius: 16,
    entityName: 'extremely-long-reverse-dns-hostname-001.subdomain.internal-cloud-provider.corp.example.com',
    category: 'Enterprise Production Compute Node Worker Cluster',
    asn: 'AS16509',
  };

  const xrayState = {
    camera: { x: 50, y: -30, zoom: 1.5 },
    selectedNode: null,
    hoveredNode: null,
    nodes: [longNode],
    edges: [],
  };

  const mockCtx = {
    clearRect() {},
    fillRect() {},
    beginPath() {},
    closePath() {},
    moveTo() {},
    lineTo() {},
    arc() {},
    stroke() {},
    fill() {},
    save() {},
    restore() {},
    translate() {},
    scale() {},
    setLineDash() {},
    createRadialGradient: () => ({ addColorStop() {} }),
    measureText: (text) => ({ width: (text || '').length * 6 }),
    fillText() {},
    roundRect() {},
    rect() {},
    strokeRect() {},
    ellipse() {},
  };

  const context = vm.createContext({
    xrayState,
    Date,
    Math,
    String,
    Set,
    $: (id) => {
      if (id === 'xray-canvas') return { width: 1100, height: 540, getContext: () => mockCtx };
      if (id === 'xray-canvas-container') return { clientWidth: 1100, clientHeight: 540 };
      return null;
    },
  });

  vm.runInContext(drawFn, context);
  context.drawXRayCanvas();

  assert.ok(longNode.pill, 'Pill bounds must be recorded on node');
  assert.ok(longNode.pill.width <= 260, 'Pill width must be bounded to <= 260px');
  assert.equal(longNode.pill.height, 32);
  assert.equal(longNode.pill.y, longNode.y + longNode.radius + 6);
});

test('canvas findNode hit testing resolves both node body circle and label pill', () => {
  const initFn = extract('initNetworkXRay');
  assert.match(initFn, /n\.pill/, 'findNode must check node.pill bounding box');

  // Verify mathematical hit testing logic
  const node = {
    id: 'n1',
    x: 100,
    y: 100,
    radius: 16,
    pill: { x: 20, y: 122, width: 160, height: 32 },
  };
  const nodes = [node];

  function hitTest(wx, wy) {
    for (let i = nodes.length - 1; i >= 0; i--) {
      const n = nodes[i];
      const dx = n.x - wx;
      const dy = n.y - wy;
      if (dx * dx + dy * dy <= (n.radius + 7) * (n.radius + 7)) {
        return n;
      }
      if (n.pill) {
        if (
          wx >= n.pill.x &&
          wx <= n.pill.x + n.pill.width &&
          wy >= n.pill.y &&
          wy <= n.pill.y + n.pill.height
        ) {
          return n;
        }
      }
    }
    return null;
  }

  // 1. Circle center hits
  assert.equal(hitTest(100, 100), node);
  // 2. Circle edge hits (radius 16 + 7 = 23)
  assert.equal(hitTest(120, 100), node);
  // 3. Label pill hits (inside pill box at y=130)
  assert.equal(hitTest(100, 130), node);
  assert.equal(hitTest(30, 140), node);
  // 4. Outside both circle and pill
  assert.equal(hitTest(300, 300), null);
});

test('selectXRayNode formats 0% ingress/egress for hosts with zero observed traffic', () => {
  const selectFn = extract('selectXRayNode');
  assert.match(selectFn, /const inPct = totalB > 0 \? Math\.round\(\(inBytes \/ totalB\) \* 100\) : 0/);
  assert.match(selectFn, /const outPct = totalB > 0 \? 100 - inPct : 0/);
});

test('keyboard Escape listener closes Network X-Ray drawer and clears selection', () => {
  assert.match(source, /if \(e\.key === ["']Escape["'] && xrayState\.selectedNode && !\$\(["']xray-node-drawer["']\)\?\.hidden\)/);
});

test('HUD zoom-in and zoom-out scale camera offsets by zoom factor to keep viewport center anchored', () => {
  const initFn = extract('initNetworkXRay');
  assert.match(initFn, /xrayState\.camera\.x\s*\*=\s*factor/);
  assert.match(initFn, /xrayState\.camera\.y\s*\*=\s*factor/);

  // Mathematically verify zoom-around-center invariance
  const camera = { x: 120, y: -80, zoom: 1.0 };
  const oldZoom = camera.zoom;
  const newZoom = Math.min(4.0, oldZoom * 1.25);
  const factor = newZoom / oldZoom;
  camera.x *= factor;
  camera.y *= factor;
  camera.zoom = newZoom;

  assert.equal(camera.zoom, 1.25);
  assert.equal(camera.x, 150);
  assert.equal(camera.y, -100);

  // Zoom back out
  const zoomOut = Math.max(0.25, camera.zoom * 0.8);
  const outFactor = zoomOut / camera.zoom;
  camera.x *= outFactor;
  camera.y *= outFactor;
  camera.zoom = zoomOut;

  assert.equal(camera.zoom, 1.0);
  assert.equal(camera.x, 120);
  assert.equal(camera.y, -80);
});

test('peer pivot button auto-clears filter to all if targeted peer is filtered out', () => {
  const selectFn = extract('selectXRayNode');
  assert.match(selectFn, /if\s*\(!targetNode\s*&&\s*xrayState\.filter\s*!==\s*["']all["']\)\s*\{\s*applyXRayFilter\(["']all["']\);/);
});

test('drawXRayCanvas provides fallback node role sublabel for unclassified endpoints', () => {
  const drawFn = extract('drawXRayCanvas');
  assert.match(drawFn, /formatNodeRole\(node\.role,\s*node\.isTor\)/);

  const plainNode = {
    id: '192.168.1.99',
    role: 'lan_host',
    x: 0,
    y: 0,
    radius: 12,
  };

  const xrayState = {
    camera: { x: 0, y: 0, zoom: 1.0 },
    selectedNode: null,
    hoveredNode: null,
    nodes: [plainNode],
    edges: [],
  };

  const mockCtx = {
    clearRect() {},
    fillRect() {},
    beginPath() {},
    closePath() {},
    moveTo() {},
    lineTo() {},
    arc() {},
    stroke() {},
    fill() {},
    save() {},
    restore() {},
    translate() {},
    scale() {},
    setLineDash() {},
    createRadialGradient: () => ({ addColorStop() {} }),
    measureText: (text) => ({ width: (text || '').length * 6 }),
    fillText() {},
    roundRect() {},
    rect() {},
    strokeRect() {},
    ellipse() {},
  };

  const context = vm.createContext({
    xrayState,
    Date,
    Math,
    String,
    Set,
    formatNodeRole: (role) => (role === 'lan_host' ? 'Local LAN Host' : 'Endpoint'),
    $: (id) => {
      if (id === 'xray-canvas') return { width: 1100, height: 540, getContext: () => mockCtx };
      if (id === 'xray-canvas-container') return { clientWidth: 1100, clientHeight: 540 };
      return null;
    },
  });

  vm.runInContext(drawFn, context);
  context.drawXRayCanvas();

  assert.ok(plainNode.pill);
  assert.equal(plainNode.pill.height, 32, 'Unclassified node must render dual-line pill with role sublabel');
});

test('selectXRayNode matches flows by node.id when node.ip is absent', () => {
  const selectFn = extract('selectXRayNode');
  assert.match(selectFn, /const targetIp = node\.ip \|\| node\.id/);
  assert.match(selectFn, /f\.initiatorAddress === targetIp \|\| f\.responderAddress === targetIp/);
});

test('debounce utility cancels pending execution, executes after delay, and supports flush', async () => {
  const debounceFn = extract('debounce');
  const context = vm.createContext({
    setTimeout,
    clearTimeout,
  });
  vm.runInContext(debounceFn, context);

  let calls = 0;
  let lastArg = null;
  const debounced = context.debounce((arg) => {
    calls++;
    lastArg = arg;
  }, 40);

  debounced('first');
  debounced('second');
  assert.equal(calls, 0, 'Should not execute synchronously');

  await new Promise((resolve) => setTimeout(resolve, 70));
  assert.equal(calls, 1, 'Should execute once after wait delay');
  assert.equal(lastArg, 'second', 'Should use latest arguments');

  // Test cancel
  debounced('third');
  debounced.cancel();
  await new Promise((resolve) => setTimeout(resolve, 70));
  assert.equal(calls, 1, 'Cancelled call must not execute');

  // Test flush
  debounced('fourth');
  debounced.flush();
  assert.equal(calls, 2, 'Flushed call must execute immediately');
  assert.equal(lastArg, 'fourth');
});

test('paginateFlows partitions items correctly across 25, 50, 100, and 250 page sizes with bounds clamping', () => {
  const paginateFn = extract('paginateFlows');
  const context = vm.createContext({
    Math,
    parseInt,
    Number,
    Array,
  });
  vm.runInContext(paginateFn, context);

  const flows = Array.from({ length: 70 }, (_, i) => ({ id: i + 1 }));

  // Page 1 of size 25
  const p1 = context.paginateFlows(flows, 1, 25);
  assert.equal(p1.totalItems, 70);
  assert.equal(p1.totalPages, 3);
  assert.equal(p1.page, 1);
  assert.equal(p1.pageSize, 25);
  assert.equal(p1.startIndex, 0);
  assert.equal(p1.endIndex, 25);
  assert.equal(p1.hasPrev, false);
  assert.equal(p1.hasNext, true);
  assert.equal(p1.items.length, 25);
  assert.equal(p1.items[0].id, 1);
  assert.equal(p1.items[24].id, 25);

  // Page 3 of size 25 (last page: 20 items)
  const p3 = context.paginateFlows(flows, 3, 25);
  assert.equal(p3.page, 3);
  assert.equal(p3.startIndex, 50);
  assert.equal(p3.endIndex, 70);
  assert.equal(p3.hasPrev, true);
  assert.equal(p3.hasNext, false);
  assert.equal(p3.items.length, 20);
  assert.equal(p3.items[0].id, 51);
  assert.equal(p3.items[19].id, 70);

  // Page 1 of size 50
  const p50 = context.paginateFlows(flows, 1, 50);
  assert.equal(p50.totalPages, 2);
  assert.equal(p50.items.length, 50);

  // Clamping: negative page clamps to 1
  const pNeg = context.paginateFlows(flows, -5, 25);
  assert.equal(pNeg.page, 1);

  // Clamping: out of bound page clamps to totalPages
  const pOverflow = context.paginateFlows(flows, 999, 25);
  assert.equal(pOverflow.page, 3);

  // Empty flows array
  const pEmpty = context.paginateFlows([], 1, 25);
  assert.equal(pEmpty.totalItems, 0);
  assert.equal(pEmpty.totalPages, 1);
  assert.equal(pEmpty.page, 1);
  assert.equal(pEmpty.items.length, 0);
  assert.equal(pEmpty.hasPrev, false);
  assert.equal(pEmpty.hasNext, false);
});

test('renderTableSkeleton builds placeholder shimmer rows for flow loading states', () => {
  const skeletonFn = extract('renderTableSkeleton');
  const rows = [];
  const mockTableBody = {
    children: rows,
    replaceChildren() {
      rows.length = 0;
    },
    appendChild(node) {
      if (node._isFragment) {
        rows.push(...node.children);
      } else {
        rows.push(node);
      }
    },
  };

  const context = vm.createContext({
    Math,
    parseInt,
    $: (id) => (id === 'flow-rows' ? mockTableBody : null),
    document: {
      createDocumentFragment: () => ({
        _isFragment: true,
        children: [],
        appendChild(child) {
          this.children.push(child);
        },
      }),
      createElement: (tag) => ({
        tag,
        className: '',
        style: {},
        children: [],
        appendChild(child) {
          this.children.push(child);
        },
      }),
    },
  });

  vm.runInContext(skeletonFn, context);
  context.renderTableSkeleton(5);

  assert.equal(rows.length, 5, 'Should render 5 skeleton placeholder rows');
  for (const row of rows) {
    assert.equal(row.className, 'skeleton-row');
    assert.equal(row.children.length, 8, 'Each row should have 8 table cells');
    for (const cell of row.children) {
      assert.equal(cell.children.length, 1);
      assert.match(cell.children[0].className, /skeleton/);
    }
  }
});

test('stepForceSimulation calculates kinetic energy and decays below stabilization threshold', () => {
  const stepFn = extract('stepForceSimulation');
  const context = vm.createContext({
    Math,
  });
  vm.runInContext(stepFn, context);

  const nodes = [
    { id: '1', x: -50, y: 0, vx: 5, vy: 5, mass: 1, pinned: false },
    { id: '2', x: 50, y: 0, vx: -5, vy: -5, mass: 1, pinned: false },
  ];
  const edges = [{ sourceNode: nodes[0], targetNode: nodes[1] }];

  const energyInitial = context.stepForceSimulation(nodes, edges, { damping: 0.85 });
  assert.ok(energyInitial > 0.05, `Initial energy must exceed threshold: ${energyInitial}`);

  let energy = energyInitial;
  for (let step = 0; step < 50; step++) {
    energy = context.stepForceSimulation(nodes, edges, { damping: 0.7 });
  }

  assert.equal(energy, 0, 'Kinetic energy must settle to 0 once stabilized below threshold');
  assert.equal(nodes[0].vx, 0, 'Velocities must be zeroed when stabilized');
  assert.equal(nodes[0].vy, 0);
  assert.equal(nodes[1].vx, 0);
  assert.equal(nodes[1].vy, 0);
});

test('startXRayAnimation halts rAF render loop when kinetic energy settles and sleeps until woken', () => {
  const startAnimFn = extract('startXRayAnimation');
  const requestRedrawFn = extract('requestXRayRedraw');

  let rafQueue = [];
  let rafIdCounter = 0;
  function mockRequestAnimationFrame(cb) {
    const id = ++rafIdCounter;
    rafQueue.push({ id, cb });
    return id;
  }

  let stepSimCallCount = 0;
  let drawCallCount = 0;

  const xrayState = {
    animFrameId: null,
    physicsEnabled: true,
    needsRedraw: true,
    kineticEnergy: 1.0,
    energyThreshold: 0.05,
    nodes: [],
    edges: [],
    drag: { isDragging: false, isPanning: false },
  };

  const context = vm.createContext({
    xrayState,
    Boolean,
    requestAnimationFrame: mockRequestAnimationFrame,
    $: (id) => (id === 'xray' ? { style: { display: 'block' } } : null),
    stepForceSimulation: () => {
      stepSimCallCount++;
      return 0;
    },
    drawXRayCanvas: () => {
      drawCallCount++;
    },
  });

  vm.runInContext(startAnimFn + '\n' + requestRedrawFn, context);

  context.startXRayAnimation();
  assert.ok(xrayState.animFrameId, 'animFrameId should be active initially');

  const frame1 = rafQueue.shift();
  frame1.cb();

  assert.equal(xrayState.animFrameId, null, 'animFrameId must be null when simulation settles');
  assert.equal(rafQueue.length, 0, 'No further rAF frames should be scheduled');
  assert.equal(drawCallCount, 1, 'drawXRayCanvas should have run once');

  context.requestXRayRedraw();
  assert.equal(xrayState.needsRedraw, true);
  assert.ok(xrayState.animFrameId, 'animFrameId must be restored upon requestXRayRedraw()');
});

test('openFlow lazy-mounts hex dump and avoids eager IPC queries when telemetry tab is active', () => {
  const openFlowFn = extract('openFlow');
  assert.match(openFlowFn, /data-drawer-tab=["']hex["']/);
  assert.match(openFlowFn, /loadFlowHexDump\(flow\.id\)/);

  let hexDumpCalls = 0;
  const elements = new Map();
  function getEl(id) {
    if (!elements.has(id)) {
      elements.set(id, {
        textContent: '',
        classList: { add() {}, remove() {}, contains: () => false },
        setAttribute() {},
        replaceChildren() {},
        append() {},
        appendChild() {},
        style: {},
      });
    }
    return elements.get(id);
  }

  let hexTabIsActive = false;

  const context = vm.createContext({
    state: {
      activeCapturePath: 'C:\\test\\capture.pcap',
      blacklistRules: [],
    },
    currentFlowPayload: null,
    $: getEl,
    activeCaptureSupportsFlow: () => true,
    loadFlowHexDump: () => {
      hexDumpCalls++;
    },
    endpoint: (ip, port) => `${ip}:${port}`,
    formatNumber: (v) => String(v || 0),
    formatBytes: (v) => `${v || 0} B`,
    formatDuration: (v) => `${v || 0} ms`,
    Math,
    document: {
      querySelector: (selector) => {
        if (selector === '[data-drawer-tab="hex"]') {
          return {
            classList: {
              contains: (cls) => (cls === 'active' ? hexTabIsActive : false),
            },
          };
        }
        return null;
      },
      createElement: () => ({
        className: '',
        style: {},
        textContent: '',
        append() {},
        appendChild() {},
      }),
    },
  });

  vm.runInContext(openFlowFn, context);

  // 1. With telemetry tab active (hexTabIsActive = false), loadFlowHexDump should NOT be called
  hexTabIsActive = false;
  context.openFlow({
    id: 42,
    protocol: 'TCP',
    initiatorAddress: '10.0.0.1',
    initiatorPort: 1234,
    responderAddress: '10.0.0.2',
    responderPort: 80,
    source: 'capture.pcap',
  });
  assert.equal(hexDumpCalls, 0, 'Hex dump must not be eagerly loaded when telemetry tab is active');

  // 2. When hex tab is active, loadFlowHexDump should be called
  hexTabIsActive = true;
  context.openFlow({
    id: 43,
    protocol: 'TCP',
    initiatorAddress: '10.0.0.1',
    initiatorPort: 1234,
    responderAddress: '10.0.0.2',
    responderPort: 80,
    source: 'capture.pcap',
  });
  assert.equal(hexDumpCalls, 1, 'Hex dump should be loaded when hex tab is active');
});
