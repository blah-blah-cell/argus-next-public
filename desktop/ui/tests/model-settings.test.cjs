const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../app.js'), 'utf8');
const html = fs.readFileSync(path.join(__dirname, '../index.html'), 'utf8');
function extract(name) {
  const start = source.search(new RegExp(`(?:async )?function ${name}\\(`));
  assert.ok(start >= 0);
  const end = source.slice(start + 1).search(/\n(?:async )?function /);
  return source.slice(start, end < 0 ? source.length : start + 1 + end);
}
function harness(invoke = async () => {throw Error('offline');}) {
  const elements = new Map();
  class Element {
    constructor(tag) {this.tag = tag; this.children = []; this.attributes = {}; this.listeners = {};}
    set id(value) {this._id = value; elements.set(value, this);}
    get id() {return this._id;}
    setAttribute(key, value) {this.attributes[key] = value;}
    removeAttribute(key) {delete this.attributes[key];}
    addEventListener(type, callback) {this.listeners[type] = callback;}
    append(...items) {this.children.push(...items);}
    replaceChildren(...items) {this.children = items;}
    querySelectorAll(tag) {return this.children.flatMap(child => [...(child.tag === tag ? [child] : []), ...child.querySelectorAll(tag)]);}
  }
  for (const id of ['model-settings-list', 'model-settings-status', 'model-settings-retry', 'model-settings-reset']) {
    const element = new Element('div'); element.id = id;
  }
  const state = {runtimePreferences: {disabledModels: ['shadow'], selectableModels: [
    {id: 'lstm', available: true, detail: 'Checkpoint present; runtime loading not checked.'},
    {id: 'shadow', available: false, detail: 'Experimental checkpoint unavailable; research only.'},
    {id: 'old', available: false, detail: 'Unavailable.'},
  ]}};
  const context = vm.createContext({state, invoke, document: {createElement: tag => new Element(tag)}, $: id => elements.get(id)});
  vm.runInContext(['modelDisplayName', 'renderModelSettings', 'saveModelSelection', 'resetModelPreferences', 'reloadModelPreferences'].map(extract).join('\n'), context);
  context.renderModelSettings();
  return {context, elements, state, controls: elements.get('model-settings-list').querySelectorAll('input')};
}
test('settings has a named section, persistent live status, and retry', () => {
  assert.match(html, /aria-labelledby="model-settings-title"/);
  assert.match(html, /id="model-settings-status" role="status" aria-live="polite"/);
  assert.match(html, /id="model-settings-retry" type="button"/);
});
test('backend catalog controls choices; unavailable opt-in stays off and disabled', () => {
  const {controls, elements} = harness();
  assert.equal(controls.length, 3);
  assert.equal(controls[0].checked, true);
  assert.equal(controls[0].disabled, false);
  assert.equal(controls[1].checked, false);
  assert.equal(controls[1].disabled, true);
  assert.equal(controls[2].checked, true);
  assert.equal(controls[2].disabled, false, 'unavailable requested model may be turned off');
  assert.match(elements.get('model-setting-1-detail').textContent, /Experimental checkpoint unavailable/);
  assert.match(controls[0].attributes['aria-describedby'], /model-settings-status/);
});
test('failed save restores preference without removing focused checkbox and exposes error', async () => {
  const {context, controls, elements, state} = harness();
  controls[0].checked = false;
  await context.saveModelSelection(controls[0], state.runtimePreferences.selectableModels[0]);
  assert.equal(controls[0].checked, true);
  assert.equal(controls[0].disabled, false);
  assert.equal(elements.get('model-settings-list').querySelectorAll('input')[0], controls[0]);
  assert.equal(controls[0].attributes['aria-invalid'], 'true');
  assert.match(elements.get('model-settings-status').textContent, /not saved: Error: offline/);
  assert.equal(elements.get('model-settings-list').attributes['aria-busy'], 'false');
});
test('successful save uses exact backend ID and authoritative returned preferences', async () => {
  let call;
  const fixture = harness(async (command, args) => {
    call = {command, args};
    return {...fixture.state.runtimePreferences, disabledModels: ['shadow', 'lstm']};
  });
  fixture.controls[0].checked = false;
  await fixture.context.saveModelSelection(fixture.controls[0], fixture.state.runtimePreferences.selectableModels[0]);
  assert.equal(call.command, 'set_model_enabled');
  assert.equal(call.args.modelId, 'lstm');
  assert.equal(call.args.enabled, false);
  assert.equal(fixture.controls[0].checked, false);
  assert.match(fixture.elements.get('model-settings-status').textContent, /future analysis segments/);
});
test('absent backend catalog fails closed with retry and no invented models', () => {
  const {context, state, elements} = harness();
  state.runtimePreferences = {};
  context.renderModelSettings();
  assert.equal(elements.get('model-settings-list').children.length, 0);
  assert.equal(elements.get('model-settings-retry').hidden, false);
  assert.match(elements.get('model-settings-status').textContent, /unavailable from this runtime/);
});
test('atomic model reset uses backend defaults and preserves unrelated settings', async () => {
  let command;
  const fixture = harness(async name => {
    command = name;
    return {...fixture.state.runtimePreferences, boostMode: true, disabledModels: ['shadow']};
  });
  fixture.state.runtimePreferences.disabledModels.push('lstm');
  await fixture.context.resetModelPreferences();
  assert.equal(command, 'reset_model_preferences');
  assert.equal(fixture.state.runtimePreferences.boostMode, true);
  const controls = fixture.elements.get('model-settings-list').querySelectorAll('input');
  assert.equal(controls[0].checked, true);
  assert.equal(controls[1].checked, false);
  assert.match(fixture.elements.get('model-settings-status').textContent, /shadow remains off/);
});
test('failed atomic reset retains authoritative choices and inline error', async () => {
  const {context, state, elements} = harness();
  const previous = state.runtimePreferences;
  await context.resetModelPreferences();
  assert.equal(state.runtimePreferences, previous);
  assert.match(elements.get('model-settings-status').textContent, /could not be restored/);
  assert.equal(elements.get('model-settings-reset').disabled, false);
});
