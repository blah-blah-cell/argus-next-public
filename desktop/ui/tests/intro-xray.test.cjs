const {test,before}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
let session,rows,mount;
before(async()=>{
  ({createIntroSession:session}=await import('../spatial/intro-session.mjs'));
  ({endpointRows:rows,mountEndpointIndex:mount}=await import('../network-xray.mjs'));
});
const read=name=>fs.readFileSync(path.join(__dirname,'..',name),'utf8');
test('introduction is first-run only, dismisses durably, and does not replay for established users',()=>{
  const values=new Map(),storage={getItem:key=>values.get(key),setItem:(key,value)=>values.set(key,value)};
  const first=session(storage);assert.equal(first.takeFirstRun(),true);assert.equal(first.takeFirstRun(),false);
  first.dismiss();assert.equal(session(storage).takeFirstRun(),false);
  values.clear();values.set('argus-onboarding-complete','argus-onboarding/v1');
  assert.equal(session(storage).takeFirstRun(),false);
  const blocked=session({getItem(){throw Error('blocked');},setItem(){throw Error('blocked');}});
  assert.equal(blocked.takeFirstRun(),false);assert.doesNotThrow(()=>blocked.dismiss());
});
test('hero starts hidden/inert and scene import is deferred to opening the introduction',()=>{
  const source=read('spatial/workspace.mjs'),html=read('index.html');
  assert.match(html,/id="spatial-workspace"[^>]*hidden inert/);
  assert.doesNotMatch(source,/^import .*createScene/m);
  assert.match(source,/async function initializeScene\(\) \{ try \{/);
  assert.match(source,/if\(!scenePromise\)scenePromise=initializeScene\(\)/);
  assert.match(source,/scene\?\.setEnabled\(introOpen&&target==='overview'&&!flat\)/);
  assert.match(source,/workspace.hidden=true;workspace.inert=true/);
  assert.match(source,/story\?\.destroy\(\);story=null/);
  assert.match(source,/if\(!disposed\)setMode\(flat\)/);
  assert.match(read('app.js'),/argus:intro-navigate.*setSettingsOpen\(false\);navigateToSection\("overview"\)/);
});
test('endpoint index filters real nodes, ranks observed volume, caps rows and handles malformed input',()=>{
  assert.deepEqual(rows(null),{total:0,rows:[]});
  const nodes=[null,{id:''},{id:'one',ip:'10.0.0.1',role:'LAN',totalInBytes:50},{id:'two',hostname:'Laptop',totalOutBytes:500},{id:'one',totalInBytes:999}, {id:'bad-count',totalInBytes:Infinity}];
  const result=rows(nodes);assert.equal(result.total,3);assert.equal(result.rows[0].id,'two');
  assert.equal(rows(nodes,'LAPTOP').rows[0].id,'two');assert.equal(rows(nodes,'10.0.0').total,1);
  assert.equal(rows(nodes,'missing').total,0);assert.equal(rows(nodes,'',1).rows.length,1);
  assert.equal(rows(Array.from({length:250},(_,i)=>({id:String(i)}))).rows.length,200);
  assert.equal(rows(nodes,'bad-count').rows[0].bytes,0);
});
test('endpoint list uses native buttons, literal text, ID-only selection and preserves focus on refresh',()=>{
  class Element {
    constructor(){this.children=[];this.dataset={};this.handlers={};this.attrs={};this.value='';}
    append(...items){this.children.push(...items);}
    replaceChildren(){this.children=[];}
    setAttribute(key,value){this.attrs[key]=value;}
    addEventListener(key,fn){this.handlers[key]=fn;}
    querySelectorAll(){return this.children.flatMap(item=>item.children);}
    focus(){root.activeElement=this;}
  }
  const ids=Object.fromEntries(['xray-endpoint-search','xray-endpoint-list','xray-index-count','xray-index-empty'].map(id=>[id,new Element()]));
  const root={getElementById:id=>ids[id],createElement:()=>new Element(),activeElement:null};
  const handlers={},sent=[],host={addEventListener:(key,fn)=>handlers[key]=fn,dispatchEvent:event=>{sent.push(event);handlers[event.type]?.(event);}};
  const previous=global.CustomEvent;global.CustomEvent=class {constructor(type,options){this.type=type;this.detail=options?.detail;}};
  try {
    mount(root,host);assert.equal(sent[0].type,'argus:xray-ready');
    const data={detail:{nodes:[{id:'real',hostname:'<b>literal</b>',totalInBytes:12}]}};
    handlers['argus:xray-data'](data);
    const button=ids['xray-endpoint-list'].children[0].children[0];
    assert.equal(button.children[0].textContent,'<b>literal</b>');
    button.focus();button.handlers.click();assert.deepEqual(sent.at(-1).detail,{id:'real'});
    handlers['argus:xray-selection']({detail:{id:'real'}});assert.equal(button.attrs['aria-pressed'],'true');
    handlers['argus:xray-data'](data);assert.equal(ids['xray-endpoint-list'].children[0].children[0],button);
    assert.equal(root.activeElement,button);
    ids['xray-endpoint-search'].value='absent';ids['xray-endpoint-search'].handlers.input();
    assert.equal(ids['xray-endpoint-list'].children.length,0);assert.equal(ids['xray-index-empty'].hidden,false);
  } finally {global.CustomEvent=previous;}
});
test('topology refresh reports failures without erasing last real data and unlocks retry',async()=>{
  const source=read('app.js'),fragment=source.slice(source.indexOf('async function loadNetworkTopology()'),source.indexOf('\nfunction requestXRayRedraw()'));
  const status={dataset:{}},refresh={},topology={nodes:[],edges:[]};
  let fail=false,calls=0;
  const context=vm.createContext({xrayState:{topology:null},$:id=>id==='xray-load-status'?status:refresh,invoke:async()=>{calls++;if(fail)throw Error('offline');return topology;},updateXRayData:value=>context.xrayState.topology=value});
  vm.runInContext(fragment,context);await context.loadNetworkTopology();
  assert.equal(status.dataset.state,'ready');assert.equal(refresh.disabled,false);
  fail=true;await context.loadNetworkTopology();
  assert.equal(status.dataset.state,'error');assert.match(status.textContent,/last loaded topology.*offline/);
  assert.equal(context.xrayState.topology,topology);assert.equal(refresh.disabled,false);
  context.xrayState.loading=true;await context.loadNetworkTopology();assert.equal(calls,2);
});
