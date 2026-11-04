const {test,before} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
let core;
before(async()=>{core=await import('../spatial/core.mjs');});
const read=name=>fs.readFileSync(path.join(__dirname,'..',name),'utf8');
const flow=(id,a,b,bytes=100,protocol='tcp')=>({id,initiatorAddress:a,responderAddress:b,originalBytes:bytes,packets:2,protocol});

test('empty and malformed inputs do not invent traffic or graph entities',()=> {
  for(const input of [[],null,undefined,[{id:'bad'}],[flow('bad','a','b',NaN)],[flow('negative','a','b',-1)]]) {
    const graph=core.observedGraph(input);
    assert.equal(graph.totalFlows,0);assert.deepEqual(graph.nodes,[]);assert.deepEqual(graph.edges,[]);
  }
});
test('observed endpoints and directional edges aggregate actual flow counters',()=> {
  const graph=core.observedGraph([flow('1','a','b',100),flow('2','a','b',30),flow('3','b','a',20)]);
  assert.equal(graph.totalFlows,3);assert.equal(graph.totalNodes,2);assert.equal(graph.shownFlows,3);
  assert.equal(graph.edges.length,2);assert.equal(graph.edges[0].bytes,130);
  assert.equal(core.endpointSummary(graph,'a').bytes,150);
  assert.equal(core.endpointSummary(graph,'a').packets,6);
  assert.equal(core.endpointSummary(graph,'a').flows,3);
  assert.equal(core.endpointSummary(graph,'missing'),null);
});
test('same-address flows count once, not twice, at the endpoint',()=> {
  const graph=core.observedGraph([flow('loop','a','a')]);
  assert.equal(graph.nodes.length,1);assert.equal(graph.nodes[0].packets,2);assert.equal(graph.nodes[0].bytes,100);
});
test('protocol selection filters data and not just its visual appearance',()=> {
  const input=[flow('1','a','b',10,'TCP'),flow('2','a','c',20,'udp')];
  assert.equal(core.observedGraph(input,'tcp').totalFlows,1);
  assert.equal(core.observedGraph(input,'udp').nodes.some(node=>node.address==='b'),false);
  assert.equal(core.observedGraph(input,'all').totalFlows,2);
});
test('deterministic layout preserves entity identity regardless of source ordering',()=> {
  const input=[flow('1','a','b'),flow('2','a','c')];
  const first=core.observedGraph(input),second=core.observedGraph([...input].reverse());
  assert.deepEqual(first,second);
  assert.deepEqual(core.endpointPosition('a'),core.endpointPosition('a'));
  assert.notDeepEqual(core.endpointPosition('a'),core.endpointPosition('b'));
  assert.ok(core.endpointPosition('a').every(Number.isFinite));
});
test('large observed networks are capped and disclose omitted flow counts',()=> {
  const input=Array.from({length:500},(_,i)=>flow(String(i),`source-${i}`,`target-${i}`,i+1));
  const graph=core.observedGraph(input);
  assert.ok(graph.nodes.length<=core.LIMITS.nodes);assert.ok(graph.shownFlows<=core.LIMITS.flows);
  assert.equal(graph.totalFlows,500);assert.equal(graph.totalNodes,1000);
  assert.equal(graph.shownFlows+graph.omittedFlows,500);
  assert.equal(graph.nodes[0].address,'source-499');
});
test('camera and render budgets remain bounded even at giant display sizes',()=> {
  assert.deepEqual(core.clampCamera({yaw:NaN,pitch:Infinity,distance:NaN}),{yaw:.3,pitch:.15,distance:11});
  assert.equal(core.clampCamera({pitch:9,distance:999}).pitch,1.2);
  assert.equal(core.clampCamera({pitch:-9,distance:1}).distance,6);
  for(const [w,h,dpr] of [[870,510,2],[3840,2160,3],[16000,10000,2]]) {
    const ratio=core.pixelRatio(w,h,dpr);
    assert.ok(w*h*ratio*ratio<=core.LIMITS.renderPixels+1);
    assert.ok(ratio<=2&&ratio>0);
  }
});
test('flow bridge sends only current state and refuses unknown endpoint inspection',()=> {
  const source=read('app.js');
  const fragment=source.slice(source.indexOf('window.addEventListener("argus:spatial-ready"'),source.indexOf('\ndocument.querySelectorAll("[data-alert-filter]"'));
  const handlers=new Map(),received=[],actions=[];
  const input={value:'',dispatchEvent(){actions.push('filter');},focus(){actions.push('focus');}};
  const state={flows:[flow('one','a','b')]};
  const context=vm.createContext({state,window:{ArgusWorkspace:{setFlows(value){received.push(value);}},addEventListener(name,handler){handlers.set(name,handler);}},
    navigateToSection(name){actions.push(name);},$(id){return id==='flow-filter'?input:null;},Event:class {},openFlow(value){actions.push(value.id);}});
  vm.runInContext(fragment,context);
  handlers.get('argus:spatial-ready')();assert.equal(received[0],state.flows);
  handlers.get('argus:spatial-inspect')({detail:{address:'invented'}});assert.deepEqual(actions,[]);
  handlers.get('argus:spatial-inspect')({detail:{address:'a'}});
  assert.deepEqual(actions,['flows','filter','focus','one']);assert.equal(input.value,'a');
});
test('fallback, keyboard-equivalent controls and endpoint list are in the shipped UI',()=> {
  const html=read('index.html'),workspace=read('spatial/workspace.mjs');
  for(const label of ['Rotate left','Rotate right','Zoom in','Zoom out','Camera view','Endpoint inspection'])assert.ok(html.includes(label));
  assert.match(html,/id="spatial-flat"/);assert.match(html,/id="spatial-endpoints"/);
  assert.match(workspace,/unavailable=true;.*setMode\(true\)/);
  assert.match(workspace,/scene\?\.setEnabled\(introOpen&&target==='overview'&&!flat\)/);
  assert.match(workspace,/textContent=node.address/);
  assert.doesNotMatch(workspace,/innerHTML|Math.random|fetch\(/);
});
test('renderer lifecycle is demand-driven, local-only and handles lost acceleration',()=> {
  const source=read('spatial/scene.mjs');
  assert.match(source,/import\('\.\.\/vendor\/three\/three.module.js'\)/);
  assert.match(source,/failIfMajorPerformanceCaveat: true/);
  assert.match(source,/IntersectionObserver/);assert.match(source,/visibilitychange/);
  assert.match(source,/webglcontextlost/);assert.match(source,/renderer.dispose\(\)/);
  assert.doesNotMatch(source,/setInterval|setAnimationLoop|Math.random/);
});

test('identity renderer cannot put endpoint meshes, IP labels or flow edges over the hero',()=> {
  const source=read('spatial/scene.mjs'),workspace=read('spatial/workspace.mjs'),html=read('index.html');
  assert.doesNotMatch(source,/IcosahedronGeometry|QuadraticBezierCurve3|dataGroup|nodeMeshes|labelRows|onSelect|graph\.nodes|graph\.edges/);
  assert.doesNotMatch(workspace,/scene\?\.update\(graph\)|scene\.update\(graph\)|scene\?\.select\(address\)/);
  assert.doesNotMatch(html,/id="spatial-labels"/);
  assert.match(source,/const mark=hero.group;world.add\(mark\)/);
});

test('flat fallback stays a single clean A even with observed flow data loaded',()=> {
  const source=read('spatial/workspace.mjs');
  const fragment=source.slice(source.indexOf('function renderFlat()'),source.indexOf('\nfunction setMode'));
  const svg={children:[],replaceChildren(){this.children=[];},append(node){this.children.push(node);}};
  const context=vm.createContext({$:id=>{assert.equal(id,'spatial-flat');return svg;},element:(name,attrs)=>({name,attrs}),graph:core.observedGraph([flow('real','a','b')])});
  vm.runInContext(fragment,context);context.renderFlat();
  assert.equal(svg.children.length,1);assert.equal(svg.children[0].name,'path');
  assert.equal(svg.children[0].attrs['fill-rule'],'evenodd');
  context.renderFlat();assert.equal(svg.children.length,1);
});

test('actual endpoint indexing and evidence access remain separate from decoration',()=> {
  const workspace=read('spatial/workspace.mjs'),app=read('app.js');
  assert.match(workspace,/graph=observedGraph\(flows/);assert.match(workspace,/list.append\(button\)/);
  assert.match(workspace,/button.addEventListener\('click',\(\)=>select\(node.address/);
  assert.match(workspace,/new CustomEvent\('argus:spatial-inspect'/);
  assert.match(app,/function drawXRayCanvas\(/);assert.match(app,/function filterTopology\(/);
});

test('scroll choreography is bounded, continuous and reversible',()=> {
  const start=core.storyPose(0),middle=core.storyPose(.7868),end=core.storyPose(1);
  assert.equal(start.chapter,'identity');assert.equal(middle.chapter,'checks');assert.equal(end.chapter,'start');
  assert.deepEqual(core.storyPose(-100),start);assert.deepEqual(core.storyPose(NaN),start);
  assert.deepEqual(core.storyPose(100),end);assert.deepEqual(core.storyPose(.7868),middle);
  let prior=start;
  for(let i=1;i<=1000;i++) {
    const pose=core.storyPose(i/1000);
    for(const key of ['x','y','rotationY','rotationZ','scale'])assert.ok(Number.isFinite(pose[key]));
    assert.ok(Math.abs(pose.x-prior.x)<.1);assert.ok(Math.abs(pose.rotationY-prior.rotationY)<.1);
    assert.ok(pose.scale>=1&&pose.scale<=1.35);prior=pose;
  }
});
test('native scroll driver does not hijack wheel input or poll page position',()=> {
  const source=read('spatial/scroll.mjs'),html=read('index.html');
  assert.match(source,/new ViewTimeline/);assert.match(source,/getComputedTiming\(\).progress/);
  assert.match(source,/prefers-reduced-motion/);assert.match(source,/attributeFilter:\['data-motion'\]/);
  assert.match(source,/static-fallback/);assert.match(source,/animation\?\.cancel/);
  assert.doesNotMatch(source,/scrollY|scrollTop|getBoundingClientRect|preventDefault|setInterval|requestAnimationFrame/);
  assert.match(html,/data-workspace-jump>Skip to setup/);
});

test('scroll driver cancels on reduced motion, respects missing support, and tears down listeners',()=> {
  function setup(supported=true) {
    const handlers=new Map(),classes=new Set(),root={dataset:{motion:'full'}},subject={dataset:{},classList:{toggle(name,on){if(on)classes.add(name);else classes.delete(name);}}};
    let changed,wakes=0,cancels=0,disconnected=false;
    const media={matches:false,addEventListener(name,fn){handlers.set('media:'+name,fn);},removeEventListener(name){handlers.delete('media:'+name);}};
    const context=vm.createContext({window:{matchMedia(){return media;}},
      document:{documentElement:root,getElementById(){return {animate(){return {effect:{getComputedTiming(){return {progress:.4};}},cancel(){cancels++;}};}};},addEventListener(name,fn){handlers.set(name,fn);},removeEventListener(name){handlers.delete(name);}},
      MutationObserver:class {constructor(fn){changed=fn;}observe(){}disconnect(){disconnected=true;}},
      ...(supported?{ViewTimeline:class {}}:{}),
    });
    vm.runInContext(read('spatial/scroll.mjs').replace('export function','function'),context);
    const driver=context.createScrollStory(subject,()=>wakes++);
    return {driver,subject,root,handlers,classes,change:()=>changed(),stats:()=>({wakes,cancels,disconnected})};
  }
  const active=setup();assert.equal(active.driver.progress(),.4);assert.equal(active.subject.dataset.scroll,'native-timeline');
  active.handlers.get('scroll')();assert.equal(active.stats().wakes,2);
  active.root.dataset.motion='reduced';active.change();
  assert.equal(active.subject.dataset.scroll,'reduced');assert.equal(active.driver.progress(),0);assert.equal(active.stats().cancels,1);
  assert.equal(active.classes.has('scroll-story'),false);
  active.driver.destroy();assert.equal(active.handlers.size,0);assert.equal(active.stats().disconnected,true);
  const unsupported=setup(false);assert.equal(unsupported.subject.dataset.scroll,'static-fallback');assert.equal(unsupported.driver.progress(),0);unsupported.driver.destroy();
});
