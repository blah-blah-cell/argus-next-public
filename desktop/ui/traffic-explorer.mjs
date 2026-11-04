// Display-only adapter for LiveHistoryView. No IPC, storage, timers or generated traffic.
const finiteCount = value => typeof value === 'number' && Number.isFinite(value) && value >= 0;
const safeCounter = value => finiteCount(value) && Number.isSafeInteger(value);
const SVG_NS = 'http://www.w3.org/2000/svg';
export const PLOT = Object.freeze({left: 46, right: 744, top: 18, bottom: 142});

export function trafficView(history, windowSeconds = 60, metric = 'packets') {
  const seconds = safeCounter(history?.segmentSeconds) && history.segmentSeconds > 0
    ? history.segmentSeconds : 5;
  const input = Array.isArray(history?.samples) ? history.samples : [];
  const valid = [];
  let omitted = 0;
  for (const sample of input) {
    if (!safeCounter(sample?.recordedAtUnixMs) || sample.recordedAtUnixMs > 8.64e15 || !safeCounter(sample?.packets) ||
        (valid.length && sample.recordedAtUnixMs <= valid.at(-1).recordedAtUnixMs)) {
      omitted++; continue;
    }
    valid.push(sample);
  }
  const latestTime = valid.at(-1)?.recordedAtUnixMs;
  const cutoff = latestTime - windowSeconds * 1000;
  const samples = windowSeconds > 0 ? valid.filter(sample => sample.recordedAtUnixMs > cutoff) : valid;
  const field = metric === 'bytes' ? 'bytes' : 'packets';
  const points = samples.map(sample => ({sample,
    rate: safeCounter(sample[field]) ? sample[field] / seconds : null}));
  const rates = points.map(point => point.rate).filter(rate => rate !== null);
  return {seconds, samples, points, omitted, metric: field,
    latestTime, firstTime: samples[0]?.recordedAtUnixMs,
    peak: rates.length ? Math.max(...rates) : null,
    average: rates.length ? rates.reduce((a, b) => a + b, 0) / rates.length : null,
    latest: points.at(-1)?.rate ?? null,
    issues: samples.filter(sample => safeCounter(sample.captureIssues) && sample.captureIssues > 0).length};
}

export function pointPositions(view) {
  const {left, right, top, bottom} = PLOT;
  const span = view.latestTime - view.firstTime;
  const max = Math.max(view.peak ?? 0, 1);
  return view.points.map(({sample, rate}) => ({sample, rate,
    x: span > 0 ? left + (sample.recordedAtUnixMs - view.firstTime) / span * (right - left) : right,
    y: rate === null ? null : bottom - rate / max * (bottom - top)}));
}

export function nearestPoint(points, x) {
  if (!points.length || !Number.isFinite(x)) return -1;
  return points.reduce((best, point, index) =>
    Math.abs(point.x - x) < Math.abs(points[best].x - x) ? index : best, 0);
}

export function chartPaths(points, seconds) {
  const groups = [];
  let group = [];
  for (const point of points) {
    if (point.y === null || (group.length &&
        point.sample.recordedAtUnixMs - group.at(-1).sample.recordedAtUnixMs > seconds * 1500)) {
      if (group.length) groups.push(group);
      group = [];
    }
    if (point.y !== null) group.push(point);
  }
  if (group.length) groups.push(group);
  return groups.map(items => {
    const line = items.map((point, index) => `${index ? 'L' : 'M'}${point.x.toFixed(2)},${point.y.toFixed(2)}`).join(' ');
    return {line, area: `${line} L${items.at(-1).x.toFixed(2)},${PLOT.bottom} L${items[0].x.toFixed(2)},${PLOT.bottom} Z`};
  });
}

function bytes(value) {
  if (!finiteCount(value)) return '—';
  const units = ['B', 'KiB', 'MiB', 'GiB'];
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit++; }
  return `${value.toLocaleString(undefined, {maximumFractionDigits: unit ? 1 : 0})} ${units[unit]}`;
}
function rate(value, metric) {
  if (value === null) return '—';
  return metric === 'bytes' ? `${bytes(value)}/s`
    : `${value.toLocaleString(undefined, {maximumFractionDigits: 1})} pkt/s`;
}
function time(value) {
  return new Date(value).toLocaleTimeString(undefined, {hour:'2-digit', minute:'2-digit', second:'2-digit'});
}
function element(tag, attrs = {}, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, String(value));
  if (text !== undefined) node.textContent = text;
  return node;
}

export function mountTrafficExplorer(root, {document: doc = root.ownerDocument} = {}) {
  const find = id => doc.getElementById(id);
  const chart = find('traffic-timeline-chart');
  const cursor = find('traffic-segment-cursor');
  if (!chart || !cursor) return null;
  const listeners = [];
  const on = (target, event, callback) => {
    target.addEventListener(event, callback);
    listeners.push(() => target.removeEventListener(event, callback));
  };
  let history, windowSeconds = 60, metric = 'packets', points = [], view;
  let selectedTime = null, crosshair, marker;
  const setText = (id, text) => { const node = find(id); if (node) node.textContent = text; };

  function inspect(index) {
    const point = points[index];
    if (!point) return;
    cursor.value = String(index);
    const sample = point.sample;
    const quality = safeCounter(sample.captureIssues)
      ? sample.captureIssues ? `${sample.captureIssues} capture issues` : 'no reported issues'
      : 'capture quality unavailable';
    const detail = `${time(sample.recordedAtUnixMs)} · ${rate(point.rate, metric)} · ${bytes(sample.bytes)} · ${quality}`;
    setText('traffic-segment-detail', detail);
    cursor.setAttribute('aria-valuetext', `Segment ${index + 1} of ${points.length}. ${detail}`);
    if (crosshair) {
      crosshair.setAttribute('x1', point.x); crosshair.setAttribute('x2', point.x);
      marker.setAttribute('cx', point.x); marker.setAttribute('cy', point.y ?? PLOT.bottom);
      marker.setAttribute('visibility', point.y === null ? 'hidden' : 'visible');
    }
  }

  function render() {
    view = trafficView(history, windowSeconds, metric);
    points = pointPositions(view);
    root.hidden = false;
    root.dataset.hasSamples = String(points.length > 0);
    const comparisonLabel = find('timeline-comparison')?.previousElementSibling;
    if (comparisonLabel) comparisonLabel.textContent = 'LAST COMPLETED SEGMENT';
    const empty = find('traffic-chart-empty');
    if (empty) empty.hidden = points.length > 0;
    cursor.disabled = !points.length;
    cursor.max = String(Math.max(0, points.length - 1));
    chart.replaceChildren();
    crosshair = null; marker = null;
    for (const fraction of [0, .5, 1]) {
      const y = PLOT.bottom - fraction * (PLOT.bottom - PLOT.top);
      chart.append(element('line', {x1:PLOT.left, x2:PLOT.right, y1:y, y2:y, class:'timeline-grid-line'}));
      if (points.length) chart.append(element('text', {x:38, y:y + 4, 'text-anchor':'end', class:'telemetry-axis'},
        metric === 'bytes' ? bytes((view.peak ?? 0) * fraction) :
          ((view.peak ?? 0) * fraction).toLocaleString(undefined, {maximumFractionDigits:0})));
    }
    if (!points.length) {
      cursor.value = '0'; cursor.removeAttribute('aria-valuetext');
      selectedTime = null;
      setText('traffic-segment-detail', 'Awaiting capture');
      setText('traffic-timeline-range', view.omitted
        ? `${view.omitted} invalid or out-of-order samples omitted. No usable history.`
        : 'Completed capture segments appear here.');
      for (const id of ['timeline-latest-rate','timeline-average-rate','timeline-peak-rate','timeline-quality']) setText(id, '—');
      find('timeline-quality')?.classList.remove('warning');
      setText('timeline-comparison', 'Awaiting enough history');
      chart.setAttribute('aria-label', 'No usable traffic samples. No activity estimated.');
      return;
    }
    for (const path of chartPaths(points, view.seconds)) {
      chart.append(element('path', {d:path.area, class:'timeline-area'}), element('path', {d:path.line, class:'timeline-line'}));
    }
    for (const point of points) {
      if (point.y !== null) chart.append(element('circle', {cx:point.x, cy:point.y, r:1.5, class:'telemetry-observed-point'}));
      if (point.sample.captureIssues > 0 && point.y !== null) {
        chart.append(element('circle', {cx:point.x, cy:point.y, r:3, class:'telemetry-issue-point'}));
      }
    }
    crosshair = element('line', {y1:PLOT.top, y2:PLOT.bottom, class:'telemetry-crosshair'});
    marker = element('circle', {r:4, class:'timeline-latest-point'});
    chart.append(crosshair, marker);
    chart.append(element('text', {x:PLOT.left, y:165, class:'telemetry-axis'}, time(view.firstTime)));
    chart.append(element('text', {x:PLOT.right, y:165, 'text-anchor':'end', class:'telemetry-axis'}, time(view.latestTime)));
    const range = windowSeconds ? `Last ${windowSeconds / 60} min` : 'All retained history';
    setText('traffic-timeline-range', `${range} · ${points.length} observed segments · ${view.seconds}s each` +
      (view.omitted ? ` · ${view.omitted} invalid/out-of-order omitted` : ''));
    setText('timeline-latest-rate', rate(view.latest, metric));
    setText('timeline-average-rate', rate(view.average, metric));
    setText('timeline-peak-rate', rate(view.peak, metric));
    const qualityKnown = view.samples.every(sample => safeCounter(sample.captureIssues));
    setText('timeline-quality', !qualityKnown ? 'Not reported' : view.issues ? `${view.issues} issue segment${view.issues === 1 ? '' : 's'}` : 'No reported issues');
    find('timeline-quality')?.classList.toggle('warning', view.issues > 0);
    const comparison = find('timeline-comparison');
    if (comparison) {
      // This is a freshness readout, not a claim that the sensor is currently running.
      comparison.previousElementSibling.textContent = 'LAST COMPLETED SEGMENT';
      comparison.textContent = time(view.latestTime);
      comparison.classList.remove('warning');
    }
    chart.setAttribute('aria-label', `${metric === 'bytes' ? 'Byte' : 'Packet'} rate across ${points.length} observed capture segments. Use Inspect segment below for individual values. Gaps are not interpolated.`);
    const selected = points.findIndex(point => point.sample.recordedAtUnixMs === selectedTime);
    if (selectedTime !== null && selected < 0) selectedTime = null;
    inspect(selected >= 0 ? selected : points.length - 1);
  }

  for (const button of root.querySelectorAll('[data-traffic-window]')) {
    on(button, 'click', () => {
      windowSeconds = Number(button.dataset.trafficWindow);
      for (const other of root.querySelectorAll('[data-traffic-window]')) other.setAttribute('aria-pressed', String(other === button));
      render();
    });
  }
  for (const button of root.querySelectorAll('[data-traffic-metric]')) {
    on(button, 'click', () => {
      metric = button.dataset.trafficMetric;
      for (const other of root.querySelectorAll('[data-traffic-metric]')) other.setAttribute('aria-pressed', String(other === button));
      render();
    });
  }
  on(cursor, 'input', () => {
    const index = Math.max(0, Math.min(points.length - 1, Number(cursor.value)));
    selectedTime = points[index]?.sample.recordedAtUnixMs ?? null;
    inspect(index);
  });
  on(cursor, 'keydown', event => {
    if (event.key === 'Escape') { selectedTime = null; inspect(points.length - 1); }
  });
  on(chart, 'pointermove', event => {
    const box = chart.getBoundingClientRect();
    if (!box.width) return;
    inspect(nearestPoint(points, (event.clientX - box.left) / box.width * 760));
  });
  on(chart, 'pointerleave', () => {
    const selected = points.findIndex(point => point.sample.recordedAtUnixMs === selectedTime);
    inspect(selected >= 0 ? selected : points.length - 1);
  });
  render();
  return {update(next) { history = next; render(); }, dispose() { for (const remove of listeners) remove(); }};
}

if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  const root = document.getElementById('traffic-timeline');
  if (root) {
    const controller = mountTrafficExplorer(root);
    if (controller) {
      window.ArgusTrafficExplorer = controller;
      window.dispatchEvent(new Event('argus:traffic-explorer-ready'));
      window.addEventListener('pagehide', () => controller.dispose(), {once:true});
    }
  }
}
