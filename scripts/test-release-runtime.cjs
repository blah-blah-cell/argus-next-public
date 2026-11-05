// Privacy/portability regressions that do not launch the app or capture traffic.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const { spawnSync } = require('node:child_process');
const root = path.resolve(__dirname, '..');
const read = (file) => fs.readFileSync(path.join(root, file), 'utf8');
const main = read('desktop/src-tauri/src/main.rs');

test('public startup cannot invoke the capture command', () => {
  const startup = main.slice(main.indexOf('fn configure_desktop('));
  assert.doesNotMatch(startup, /start_rolling_capture_blocking\s*\(/);
  assert.doesNotMatch(startup, /start_rolling_capture_inner\s*\(/);
  assert.match(main, /async fn start_rolling_capture\(/);
});

test('public defaults do not discover private sibling runtimes or data', () => {
  assert.doesNotMatch(main, /argus-dev/);
  assert.match(main, /workspace_root\(\)\.join\("python\/\.venv\/Scripts\/python\.exe"\)/);
  assert.match(main, /workspace_root\(\)\.join\("models"\)/);
  assert.doesNotMatch(main, /join\("ArgusNext(?:\/captures)?"\)/);
  assert.match(main, /join\("ArgusNextPublic"\)/);
  assert.match(main, /join\("ArgusNextPublic\/captures"\)/);
  assert.doesNotMatch(read('crates/argus-inference/src/lib.rs'), /argus-dev/);
  assert.doesNotMatch(read('scripts/setup-windows-model-runtime.ps1'), /argus-dev/);
});

test('WebView identity, elevated task and shortcut are isolated', () => {
  const config = JSON.parse(read('desktop/src-tauri/tauri.conf.json'));
  assert.equal(config.identifier, 'dev.argus.next.public');
  assert.match(main, /SENSOR_TASK_NAME: &str = "ArgusNextPublicSensor"/);
  assert.match(read('native/pktmon-stream/install-task.ps1'), /'ArgusNextPublicSensor'/);
  assert.match(read('native/pktmon-stream/main.cpp'), /L"ArgusNextPublicLive"/);
  assert.match(read('desktop/src-tauri/windows/ensure-shortcut.ps1'), /'Argus Next Public\.lnk'/);
});

test('release resource discovery embeds no developer checkout path', () => {
  assert.doesNotMatch(main, /env!\("CARGO_MANIFEST_DIR"\)/);
  const inference = read('crates/argus-inference/src/lib.rs').split('#[cfg(test)]')[0];
  assert.doesNotMatch(inference, /env!\("CARGO_MANIFEST_DIR"\)/);
  const discovery = main.slice(main.indexOf('fn workspace_root()'), main.indexOf('fn development_root()'));
  assert.match(discovery, /current_exe\(\)/);
  assert.doesNotMatch(discovery, /current_dir\(\)/);
  assert.match(discovery, /directory\.to_path_buf\(\)/);
});

test('portable packaging retains the embedded Three.js license notice', () => {
  assert.match(read('scripts/package-windows.ps1'), /'desktop\/ui\/vendor\/three\/LICENSE' = 'THIRD_PARTY\/three-LICENSE\.txt'/);
  assert.match(read('desktop/ui/vendor/three/LICENSE'), /The MIT License/);
});

test('installer smoke test cannot erase an existing install directory', { skip: process.platform !== 'win32' }, () => {
  const script = path.join(root, 'scripts/test-installer.ps1');
  // The setup path only needs to exist: the guard must run before executing it.
  const result = spawnSync('powershell.exe', [
    '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
    '-File', script, '-SetupPath', script, '-InstallDirectory', root,
  ], { encoding: 'utf8' });
  assert.equal(result.error, undefined);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Install directory already exists/);
  assert.doesNotMatch(read('scripts/test-installer.ps1'), /Remove-Item -LiteralPath \$InstallDirectory/);
});

test('installer smoke test protects shared per-user installation registration', () => {
  const smoke = read('scripts/test-installer.ps1');
  assert.match(smoke, /HKCU:.*Uninstall\\ArgusNextPublic/);
  assert.match(smoke, /use a clean test account/);
  assert.ok(smoke.indexOf('foreach ($sharedPath') < smoke.indexOf('& $setup --verify'));
});
