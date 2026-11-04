const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../app.js'), 'utf8');
const html = fs.readFileSync(path.join(__dirname, '../index.html'), 'utf8');
const styles = fs.readFileSync(path.join(__dirname, '../styles.css'), 'utf8');
function cssRule(selector) {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const matches = [...styles.matchAll(new RegExp(`(?:^|\\n)${escaped}\\s*\\{([^}]+)\\}`, 'g'))];
  assert.ok(matches.length, `Missing CSS rule ${selector}`);
  return matches.map(match => match[1]).join('\n');
}

test('settings helper text stays readable and wraps instead of truncating', () => {
  for (const selector of ['.settings-user-card small', '.settings-select-row small',
    '.settings-panel .shutdown-preference small', '.settings-setup-button small', '.settings-privacy']) {
    const rule = cssRule(selector);
    assert.match(rule, /font-size:\s*12px/);
    assert.match(rule, /line-height:\s*1\.5/);
    assert.doesNotMatch(rule, /text-overflow:\s*ellipsis|white-space:\s*nowrap/);
  }
});

test('narrow settings stacks select controls without shrinking explanatory text', () => {
  assert.match(styles, /@media\s*\(max-width:\s*520px\)\s*\{\s*\.settings-select-row\s*\{\s*grid-template-columns:\s*minmax\(0,\s*1fr\)/);
  assert.match(cssRule('.settings-select-row select'), /max-width:\s*100%/);
  assert.match(cssRule('.settings-panel .shutdown-preference > span'), /min-width:\s*0/);
});

function extract(name) {
  const start = source.indexOf(`function ${name}(`);
  assert.ok(start >= 0, `Missing function ${name}`);
  const end = source.indexOf('\nfunction ', start + 1);
  return source.slice(start, end < 0 ? source.length : end);
}

test('settings uses a native modal while onboarding is an inline document section', () => {
  assert.match(html, /<dialog\s+class="settings-panel"/);
  assert.match(html, /<section\s+class="onboarding-dialog"/);
  assert.match(extract('setSettingsOpen'), /\.showModal\(/);
  assert.doesNotMatch(extract('showOnboarding'), /\.showModal\(/);
  assert.match(extract('showOnboarding'), /argus:journey-open/);
  assert.match(extract('closeOnboarding'), /ArgusWorkspace\?\.closeIntroduction/);
  assert.match(styles, /\.settings-panel::backdrop/);
});

test('collapsed settings keeps a name and current navigation is exposed', () => {
  assert.match(html, /id="user-settings-button"[\s\S]*?aria-label="User settings"/);
  assert.match(html, /data-target="overview"[^>]*aria-current="location"/);
  assert.match(html, /id="notice"[^>]*role="status"/);
});

test('settings open/close uses native containment and retains synchronized control state', () => {
  const attrs = new Map();
  const panel = {
    hidden: true, open: false,
    showModal() { this.open = true; },
    close() { this.open = false; },
    setAttribute(key, value) { attrs.set(`panel:${key}`, value); },
  };
  const trigger = { classList: { toggle() {} }, setAttribute(key, value) { attrs.set(key, value); } };
  const scrim = { hidden: true };
  const context = vm.createContext({
    $(id) { return { 'user-settings-panel': panel, 'settings-scrim': scrim, 'user-settings-button': trigger }[id]; },
    setTheme() {}, getStoredTheme() { return 'light'; },
  });
  vm.runInContext(extract('setSettingsOpen'), context);
  context.setSettingsOpen(true);
  assert.equal(panel.open, true);
  assert.equal(panel.hidden, false);
  assert.equal(scrim.hidden, true);
  assert.equal(attrs.get('aria-expanded'), 'true');
  context.setSettingsOpen(false);
  assert.equal(panel.open, false);
  assert.equal(panel.hidden, true);
  assert.equal(attrs.get('aria-expanded'), 'false');
});

test('closed flow inspector is inert in markup and after closing', () => {
  assert.match(html, /id="flow-drawer"[^>]*inert/);
  const drawer = { inert: false, classList: { remove() {} }, setAttribute() {} };
  const scrim = { classList: { remove() {} } };
  const context = vm.createContext({ $(id) { return id === 'flow-drawer' ? drawer : scrim; } });
  vm.runInContext(extract('closeFlow'), context);
  context.closeFlow();
  assert.equal(drawer.inert, true);
  assert.match(extract('openFlow'), /\.inert = false/);
});

test('navigation honors motion preference and updates current-section semantics', () => {
  let scrolled;
  const target = { id: 'diagnostics', style: {}, scrollIntoView(options) { scrolled = options; } };
  const links = ['overview', 'diagnostics'].map((name) => ({
    dataset: { target: name }, classList: { toggle() {} }, attributes: {},
    setAttribute(key, value) { this.attributes[key] = value; },
    removeAttribute(key) { delete this.attributes[key]; },
  }));
  const context = vm.createContext({
    state: { uiPreferences: { reduceMotion: true } },
    $(id) { return id === 'diagnostics' ? target : null; },
    document: { querySelectorAll() { return links; } },
  });
  vm.runInContext(extract('navigateToSection'), context);
  assert.equal(context.navigateToSection('diagnostics'), true);
  assert.equal(scrolled.behavior, 'instant');
  assert.equal(links[1].attributes['aria-current'], 'location');
  assert.equal(links[0].attributes['aria-current'], undefined);
  context.state.uiPreferences.reduceMotion = false;
  context.navigateToSection('diagnostics');
  assert.equal(scrolled.behavior, 'smooth');
  assert.equal(context.navigateToSection('missing'), false);
});

test('initialization failure ends pending states without claiming the sensor stopped', () => {
  const nodes = new Map([
    ['capture-state', { textContent: 'Sensor checking…' }],
    ['capture-dot', { className: 'status-dot' }],
    ['model-state', { textContent: 'Checking…' }],
    ['moe-state', { textContent: 'MoE status pending' }],
    ['diagnostics-state', { textContent: 'CHECKING' }],
    ['learning-state', { textContent: 'CHECKING' }],
    ['diagnostics-grid', { textContent: 'Running local checks...' }],
  ]);
  const notices = [];
  const context = vm.createContext({
    $(id) { return nodes.get(id); },
    window: {}, notice(text, kind) { notices.push({ text, kind }); },
  });
  vm.runInContext(extract('renderInitializationFailure'), context);
  context.renderInitializationFailure(new Error('test IPC failure'));
  assert.equal(nodes.get('capture-state').textContent, 'Sensor status unavailable');
  assert.equal(nodes.get('diagnostics-state').textContent, 'NOT VERIFIED');
  assert.equal(nodes.get('learning-state').textContent, 'UNAVAILABLE');
  assert.match(notices[0].text, /desktop app/i);
  assert.equal(notices[0].kind, 'error');
  assert.doesNotMatch(nodes.get('capture-state').textContent, /stopped|running/i);
});

test('a partial initialization failure does not erase already verified component status', () => {
  const nodes = new Map([
    ['capture-state', { textContent: 'Live · 200 packets · 0 issues' }],
    ['model-state', { textContent: 'ONNX CPU' }],
    ['moe-state', { textContent: 'Verified router' }],
    ['diagnostics-state', { textContent: 'HEALTHY' }],
    ['learning-state', { textContent: 'IDLE' }],
  ]);
  const context = vm.createContext({ $(id) { return nodes.get(id); }, window: {}, notice() {} });
  vm.runInContext(extract('renderInitializationFailure'), context);
  context.renderInitializationFailure(new Error('other component failed'));
  assert.equal(nodes.get('capture-state').textContent, 'Live · 200 packets · 0 issues');
  assert.equal(nodes.get('model-state').textContent, 'ONNX CPU');
  assert.equal(nodes.get('diagnostics-state').textContent, 'HEALTHY');
});
