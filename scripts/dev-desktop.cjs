// One development entry point; no release build or generated UI files.
const fs = require('node:fs');
const path = require('node:path');
const {spawn,spawnSync} = require('node:child_process');
const {startDevServer,HOST,PORT} = require('./desktop-dev-server.cjs');
const projectRoot = path.resolve(__dirname,'..');
function launchPlan(root=projectRoot) {
  return {
    command:'cargo', args:['tauri','dev','--config','src-tauri/tauri.dev.conf.json'],
    cwd:path.join(root,'desktop'),
  };
}
function preflight() {
  if (Number(process.versions.node.split('.')[0]) < 22) throw Error('Node.js 22+ is required');
  if (process.platform !== 'win32') throw Error('This native development launcher currently supports Windows only');
  const probe = spawnSync('powershell.exe',['-NoProfile','-NonInteractive','-Command',
    "$ErrorActionPreference = 'Stop'; @(Get-Process -Name argus-desktop -ErrorAction SilentlyContinue).Count"],{encoding:'utf8',windowsHide:true});
  if (probe.error || probe.status !== 0 || !/^\d+$/.test(probe.stdout.trim())) {
    throw Error('Could not verify desktop process state; refusing to launch another native instance');
  }
  if (Number(probe.stdout.trim()) !== 0) {
    throw Error('Argus desktop is already running. Quit it through its tray menu before starting development; this launcher never stops it.');
  }
  const cli = spawnSync('cargo',['tauri','--version'],{encoding:'utf8',windowsHide:true});
  if (cli.error || cli.status !== 0) throw Error('cargo-tauri is required; see docs/DEVELOPMENT.md (nothing is installed automatically)');
  const config = JSON.parse(fs.readFileSync(path.join(projectRoot,'desktop/src-tauri/tauri.dev.conf.json'),'utf8'));
  if (config.build.devUrl !== `http://${HOST}:${PORT}`) throw Error('Dev config URL does not match the loopback server');
}
async function main() {
  preflight();
  const development = await startDevServer();
  const plan = launchPlan();
  console.log(`Authored UI: ${development.url} (loopback only; edits reload the WebView)`);
  console.log('Starting real Tauri debug shell. Native startup/IPC behavior is unchanged, including normal sensor startup.');
  const child = spawn(plan.command,plan.args,{cwd:plan.cwd,stdio:'inherit',windowsHide:true});
  // cargo-tauri owns its native child; Ctrl+C reaches the whole foreground group.
  // Do not kill desktop processes by name: an unrelated release must stay untouched.
  const stop = () => { void development.close().catch(console.error); };
  process.once('SIGINT',stop);
  process.once('SIGTERM',stop);
  try {
    process.exitCode = await new Promise((resolve,reject) => {
      child.once('error',reject);
      child.once('exit',(code,signal) => resolve(code ?? (signal ? 1 : 0)));
    });
  } finally {
    process.removeListener('SIGINT',stop);
    process.removeListener('SIGTERM',stop);
    await development.close();
  }
}
module.exports = {launchPlan,preflight};
if (require.main === module) main().catch(error => {
  console.error(`Desktop development failed: ${error.message}`); process.exitCode = 1;
});
