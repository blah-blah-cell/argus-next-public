const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const workflow = fs.readFileSync(path.join(__dirname, '../.github/workflows/release.yml'), 'utf8');

test('setup artifact path is consistent across build, smoke, attestation and publish', () => {
  const file = 'dist/setup/ArgusNextPublic-windows-x64-setup.exe';
  assert.ok(workflow.split(file).length - 1 >= 5);
  assert.doesNotMatch(workflow, /dist\/ArgusNextPublic-windows-x64-setup\.exe/);
});

test('binaries are signed before packaging, while the setup stays unsigned', () => {
  const at = (name) => workflow.indexOf(`name: ${name}`);
  const chain = [at('Build release binaries'), at('Optionally sign release binaries'),
    at('Build portable package'), at('Build setup program'),
    at('Remove temporary signing key'), at('Bundle portable package'),
    at('Attest build provenance'), at('Publish release')];
  assert.ok(chain.every(n => n >= 0));
  assert.deepEqual(chain, [...chain].sort((a,b) => a-b));
  assert.doesNotMatch(workflow, /signtool[^\n]*sign[^\n]*setup\.exe/i);
  assert.match(workflow, /certificate-aware parsing/);
});

test('distribution includes model terms and does not claim payload opt-in', () => {
  const root = path.join(__dirname, '..');
  const pack = fs.readFileSync(path.join(__dirname, 'package-windows.ps1'), 'utf8');
  const privacy = fs.readFileSync(path.join(root, 'docs/adr/0003-metadata-first-capture-and-retention.md'), 'utf8');
  assert.match(pack, /'models\/NOTICE\.md'\s*=\s*'models\/NOTICE\.md'/);
  assert.match(pack, /'NOTICE'\s*=\s*'NOTICE'/);
  assert.match(privacy, /not currently a\s+separate payload-retention opt-in/);
});

test('manual validation cannot publish a release', () => {
  assert.match(workflow, /workflow_dispatch:/);
  assert.match(workflow, /name: Upload validation artifacts\s+if: github\.event_name == 'workflow_dispatch'/);
  assert.match(workflow, /name: Publish release\s+if: github\.event_name == 'push'/);
});

test('SBOM destination exists before the scanner writes to it', () => {
  assert.match(workflow, /name: Prepare SBOM output directory[\s\S]*New-Item -ItemType Directory -Force -Path dist\/sbom[\s\S]*name: Software bill of materials/);
});
