const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {discoverSources,discoverTests,validateHtml}=require('../../../scripts/validate-desktop-ui.cjs');
test('frontend gate discovers new spatial modules and every test automatically',()=>{
  const sources=discoverSources(),tests=discoverTests();
  assert.ok(sources.some(file=>file.endsWith('hero-material-controls.mjs')));
  assert.ok(sources.some(file=>file.endsWith('app.js')));
  assert.ok(sources.every(file=>!file.includes(`${path.sep}vendor${path.sep}`)));
  assert.ok(tests.some(file=>file.endsWith('frontend-validation.test.cjs')));
  const actual=fs.readdirSync(__dirname).filter(name=>name.endsWith('.test.cjs')).length;
  assert.equal(tests.length,actual);
});
test('frontend gate rejects duplicate HTML ids',()=>{
  assert.doesNotThrow(()=>validateHtml('<div id="one"></div><div id="two"></div>'));
  assert.throws(()=>validateHtml('<div id="one"></div><div id="one"></div>'),/Duplicate HTML id/);
});
test('package invokes the complete frontend gate',()=>{
  const pkg=JSON.parse(fs.readFileSync(path.join(__dirname,'../package.json'),'utf8'));
  assert.match(pkg.scripts.validate,/validate-desktop-ui\.cjs/);
});
