const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

test('first launch does not opt in to automatic network capture', () => {
  const source = fs.readFileSync(path.join(__dirname, '../app.js'), 'utf8');
  assert.match(source, /userStoppedRolling:\s*true/);
  const guard = source.slice(source.indexOf('async function ensureRollingCapture()'), source.indexOf('function capturesReadyForAnalysis'));
  assert.match(guard, /if \(state\.userStoppedRolling \|\|/);
});
