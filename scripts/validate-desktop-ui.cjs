// Dependency-free development gate; never bundled into the running application.
const fs = require('node:fs');
const path = require('node:path');
const {spawnSync} = require('node:child_process');
const uiRoot = path.resolve(__dirname, '../desktop/ui');

function discoverSources(root=uiRoot) {
  const files=[];
  function visit(folder) {
    for (const entry of fs.readdirSync(folder,{withFileTypes:true})) {
      if (entry.isSymbolicLink()) continue;
      const file=path.join(folder,entry.name);
      if (entry.isDirectory()) {
        if (!['vendor','tests','node_modules'].includes(entry.name)) visit(file);
      } else if (/\.(?:mjs|js|cjs)$/.test(entry.name)) files.push(file);
    }
  }
  visit(root);return files.sort();
}
function discoverTests(root=uiRoot) {
  return fs.readdirSync(path.join(root,'tests')).filter(name=>name.endsWith('.test.cjs'))
    .sort().map(name=>path.join(root,'tests',name));
}
function validateHtml(html) {
  const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(match=>match[1]);
  const seen=new Set();
  for (const id of ids) {if (seen.has(id)) throw Error(`Duplicate HTML id: ${id}`);seen.add(id);}
}
function run(args) {
  const result=spawnSync(process.execPath,args,{cwd:uiRoot,stdio:'inherit'});
  if(result.error)throw result.error;
  if(result.status!==0)throw Error(`Node check failed (${result.signal||result.status}): ${args.join(' ')}`);
}
function main() {
  if(Number(process.versions.node.split('.')[0])<22)throw Error('Node.js 22+ is required for UI validation');
  const sources=discoverSources(),tests=discoverTests();
  if(!sources.length||!tests.length)throw Error('Missing frontend source or tests; refusing an empty pass');
  validateHtml(fs.readFileSync(path.join(uiRoot,'index.html'),'utf8'));
  for(const file of [__filename,...sources,...tests])run(['--check',file]);
  console.log(`UI validation: ${sources.length} authored modules, ${tests.length} test files; vendored code excluded.`);
  run(['--test',...tests]);
  console.log('VERDICT: PASS');
}
module.exports={discoverSources,discoverTests,validateHtml};
if(require.main===module) {
  try{main();}catch(error){console.error(`VERDICT: FAIL -- ${error.message}`);process.exitCode=1;}
}
