const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const modulePath = path.join(__dirname, '../traffic-explorer.mjs');
const load = () => import(pathToFileURL(modulePath));
const sample = (second, packets = 50, extra = {}) => ({
  recordedAtUnixMs: 1700000000000 + second * 1000, packets, bytes: packets * 120,
  flows: 2, captureIssues: 0, ...extra,
});

test('traffic windows use observed timestamps, not assumed contiguous samples', async () => {
  const {trafficView, pointPositions} = await load();
  const history = {segmentSeconds:5, samples:[sample(0), sample(5), sample(30), sample(65)]};
  const minute = trafficView(history,60);
  assert.deepEqual(minute.samples.map(s => s.recordedAtUnixMs), history.samples.slice(2).map(s => s.recordedAtUnixMs));
  const all = trafficView(history,0);
  const points = pointPositions(all);
  assert.ok(points[2].x - points[1].x > 4 * (points[1].x - points[0].x));
  assert.equal(all.latest,10);
  assert.equal(trafficView(history,300,'bytes').latest,1200);
});

test('invalid counters, rollback and duplicates are omitted explicitly, with no sorting or coercion', async () => {
  const {trafficView} = await load();
  const view = trafficView({samples:[sample(1),sample(0),sample(1),sample(2,NaN), sample(3,-1),
    sample(4,'50'),sample(5,Infinity),sample(6,Number.MAX_SAFE_INTEGER+1),sample(7)]},0);
  assert.equal(view.omitted,7);
  assert.equal(view.samples.length,2);
  assert.equal(view.latest,10);
  assert.equal(trafficView({samples:[sample(1,5,{recordedAtUnixMs:9e15})]}).samples.length,0);
});

test('empty, idle, missing byte and missing quality states never invent positive measurements', async () => {
  const {trafficView, pointPositions} = await load();
  assert.equal(trafficView(null).latest,null);
  assert.equal(trafficView({samples:[]}).average,null);
  const idle = trafficView({segmentSeconds:0,samples:[sample(1,0)]});
  assert.equal(idle.latest,0);
  assert.equal(idle.seconds,5);
  assert.equal(pointPositions(idle)[0].y,142);
  const unknown = trafficView({samples:[sample(1,4,{bytes:undefined,captureIssues:undefined})]},0,'bytes');
  assert.equal(unknown.latest,null);
  assert.equal(unknown.average,null);
  assert.equal(pointPositions(unknown)[0].y,null);
});

test('paths break at missing intervals/byte fields and every coordinate stays finite', async () => {
  const {trafficView,pointPositions,chartPaths,nearestPoint} = await load();
  const view = trafficView({samples:[sample(0),sample(5),sample(30),sample(35)]},0);
  const points=pointPositions(view);
  assert.equal(chartPaths(points,view.seconds).length,2);
  assert.ok(chartPaths(points,view.seconds).every(p => !/NaN|Infinity/.test(p.area+p.line)));
  assert.equal(nearestPoint(points,-100),0);
  assert.equal(nearestPoint(points,10000),3);
  assert.equal(nearestPoint([],1),-1);
  assert.equal(nearestPoint(points,NaN),-1);
  const bytesView = trafficView({samples:[sample(0),sample(5,5,{bytes:null}),sample(10)]},0,'bytes');
  assert.equal(chartPaths(pointPositions(bytesView),5).length,2);
});

class Node {
  constructor(tag='div') { this.tag=tag;this.children=[];this.attrs={};this.dataset={};this.handlers=new Map();this.textContent='';this.hidden=false;this.value='0';this.disabled=false;
    this.classes=new Set();this.classList={remove:name=>this.classes.delete(name),toggle:(name,on)=>on?this.classes.add(name):this.classes.delete(name)}; }
  setAttribute(name,value) {this.attrs[name]=String(value);}
  removeAttribute(name) {delete this.attrs[name];}
  append(...nodes) {this.children.push(...nodes);}
  replaceChildren(...nodes) {this.children=nodes;}
  addEventListener(event,fn) {if(!this.handlers.has(event))this.handlers.set(event,new Set());this.handlers.get(event).add(fn);}
  removeEventListener(event,fn) {this.handlers.get(event)?.delete(fn);}
  emit(event,detail={}) {for(const fn of this.handlers.get(event)||[])fn(detail);}
  getBoundingClientRect() {return {left:0,width:760};}
}
function fixture() {
  const ids=['traffic-timeline-chart','traffic-segment-cursor','traffic-chart-empty','traffic-segment-detail',
    'traffic-timeline-range','timeline-latest-rate','timeline-average-rate','timeline-peak-rate','timeline-quality','timeline-comparison'];
  const nodes=Object.fromEntries(ids.map(id=>[id,new Node()]));
  nodes['timeline-comparison'].previousElementSibling=new Node();
  const windows=[60,300,0].map(value=>{const n=new Node('button');n.dataset.trafficWindow=String(value);return n;});
  const metrics=['packets','bytes'].map(value=>{const n=new Node('button');n.dataset.trafficMetric=value;return n;});
  const root=new Node('section');
  const doc={getElementById:id=>nodes[id],createElementNS:(_ns,tag)=>new Node(tag)};
  root.ownerDocument=doc;
  root.querySelectorAll=selector=>selector==='[data-traffic-window]'?windows:metrics;
  return {nodes,root,doc,windows,metrics};
}

test('controller supports keyboard range inspection, metric/window selection, updates, empty reset and disposal', async () => {
  const {mountTrafficExplorer} = await load();
  const {root,doc,nodes,windows,metrics}=fixture();
  global.document=doc;
  try {
    const controller=mountTrafficExplorer(root);
    const cursor=nodes['traffic-segment-cursor'];
    assert.equal(cursor.disabled,true);
    assert.equal(nodes['timeline-latest-rate'].textContent,'—');
    controller.update({segmentSeconds:5,samples:[sample(0,50),sample(5,100),sample(10,200,{captureIssues:2})]});
    assert.equal(cursor.disabled,false);
    assert.equal(cursor.max,'2');
    assert.match(nodes['timeline-latest-rate'].textContent,/40 pkt\/s/);
    assert.equal(nodes['timeline-quality'].classes.has('warning'),true);
    cursor.value='0';cursor.emit('input');
    assert.match(cursor.attrs['aria-valuetext'],/Segment 1 of 3/);
    metrics[1].emit('click');
    assert.equal(metrics[1].attrs['aria-pressed'],'true');
    assert.match(cursor.attrs['aria-valuetext'],/Segment 1 of 3/);
    assert.match(nodes['timeline-latest-rate'].textContent,/KiB\/s/);
    windows[2].emit('click');
    assert.match(nodes['traffic-timeline-range'].textContent,/All retained/);
    controller.update({segmentSeconds:5,samples:[sample(0),sample(5),sample(10),sample(15)]});
    assert.match(cursor.attrs['aria-valuetext'],/Segment 1 of 4/);
    cursor.emit('keydown',{key:'Escape'});
    assert.match(cursor.attrs['aria-valuetext'],/Segment 4 of 4/);
    nodes['traffic-timeline-chart'].emit('pointermove',{clientX:46});
    assert.match(cursor.attrs['aria-valuetext'],/Segment 1 of 4/);
    nodes['traffic-timeline-chart'].emit('pointerleave');
    assert.match(cursor.attrs['aria-valuetext'],/Segment 4 of 4/);
    controller.update(null);
    assert.equal(cursor.disabled,true);
    assert.equal(cursor.attrs['aria-valuetext'],undefined);
    assert.equal(nodes['timeline-quality'].classes.has('warning'),false);
    assert.equal(nodes['traffic-chart-empty'].hidden,false);
    controller.dispose();
    assert.ok([...cursor.handlers.values()].every(set=>set.size===0));
  } finally {delete global.document;}
});

test('native packet history drives the display; onboarding and runtime remain separate', () => {
  const source=fs.readFileSync(modulePath,'utf8');
  const app=fs.readFileSync(path.join(__dirname,'../app.js'),'utf8');
  const html=fs.readFileSync(path.join(__dirname,'../index.html'),'utf8');
  const css=fs.readFileSync(path.join(__dirname,'../observatory.css'),'utf8');
  assert.doesNotMatch(source,/Math\.random|setInterval|requestAnimationFrame|\.invoke\(|localStorage/);
  assert.match(app,/state\.liveHistory = history/);
  assert.match(app,/ArgusTrafficExplorer\.update\(history\)/);
  assert.match(app,/argus:traffic-explorer-ready/);
  assert.match(html,/<label for="traffic-segment-cursor">/);
  assert.match(html,/id="traffic-segment-cursor"[^>]*aria-describedby="traffic-segment-detail"/);
  assert.match(css,/\.spatial-app:not\(\.on-landing\)/);
  assert.match(css,/\[data-theme="light"\]/);
  assert.match(css,/prefers-reduced-motion/);
  assert.doesNotMatch(css,/transition:\s*all|@keyframes/);
});
