const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ui = path.join(__dirname, '..');
const native = path.resolve(ui, '../src-tauri');
const read = p => fs.readFileSync(p, 'utf8');

test('approved offset-A uses identical geometry in logo and native/UI icons', () => {
  const master = read(path.join(native, 'app-icon.svg'));
  const mark = read(path.join(ui, 'branding/argus-mark.svg'));
  assert.equal(read(path.join(ui, 'branding/argus-icon.svg')), master);
  const paths = text => [...text.matchAll(/<path d="([^"]+)"/g)].map(m => m[1]);
  assert.equal(paths(master).length, 3);
  assert.deepEqual(paths(master), paths(mark));
  assert.doesNotMatch(master, /<circle|<ellipse|<image|<script|<linearGradient/);
  assert.match(master, /fill="#699c9b"/);
});

test('header and favicon use local branding without duplicate accessible text', () => {
  const html = read(path.join(ui, 'index.html'));
  assert.match(html, /rel="icon"[^>]+href="branding\/argus-icon.svg"/);
  assert.match(html, /class="brand-mark" aria-hidden="true"><img src="branding\/argus-icon.svg" alt=""/);
  const config = JSON.parse(read(path.join(native, 'tauri.conf.json')));
  assert.ok(config.bundle.icon.includes('icons/icon.ico'));
});

test('native PNG and ICO assets have expected resolutions and transparency', () => {
  for (const [name, size] of [['32x32.png',32], ['128x128.png',128], ['128x128@2x.png',256]]) {
    const png = fs.readFileSync(path.join(native, 'icons',name));
    assert.equal(png.subarray(0,8).toString('hex'),'89504e470d0a1a0a');
    assert.equal(png.readUInt32BE(16),size);
    assert.equal(png.readUInt32BE(20),size);
    assert.equal(png[25],6, 'RGBA required');
  }
  const ico = fs.readFileSync(path.join(native, 'icons/icon.ico'));
  assert.equal(ico.readUInt16LE(2),1);
  const count = ico.readUInt16LE(4);
  const sizes = Array.from({length:count},(_,i)=>ico[6+16*i]||256);
  assert.ok(sizes.includes(16));
  assert.ok(sizes.includes(32));
  assert.ok(sizes.includes(256));
});
