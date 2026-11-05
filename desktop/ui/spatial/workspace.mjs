import { observedGraph, endpointSummary } from './core.mjs';
import {bindHeroRoughness} from './hero-material-controls.mjs?v=roughness-2';
import { createScrollStory } from './scroll.mjs?v=cinematic-1';
import {createSetupStage} from './setup-stage.mjs?v=room-orbit-1';

const $ = id => document.getElementById(id);
const stage = $('spatial-stage');
const svgNS = 'http://www.w3.org/2000/svg';
let scene = null, materialControls=null, flows = [], graph = observedGraph([]), address = null;
let flat = false, unavailable = false, page = 'overview';
const workspace=$('spatial-workspace');
let story=null,scenePromise=null,introOpen=false,disposed=false;
let storage;try{storage=window.localStorage;}catch{ /* Optional local preferences. */ }
const setup=$('onboarding-dialog'),visualStory=$('intro-visual-story');
workspace.append(setup);
const setupStage=createSetupStage({setup,pin:document.querySelector('.spatial-pin'),subject:visualStory,workspace,
  onStep:(step,chapter)=>{
    const enter=document.querySelector('.spatial-enter');
    if(enter?.firstChild)enter.firstChild.textContent=step<0?['Explore Argus ','Continue to evidence ','Set up this device '][chapter]:step===0?'Continue to device checks ':'Continue to start ';
    if(step>=0)window.dispatchEvent(new CustomEvent('argus:setup-stage',{detail:{step}}));
  }});
const overview = new Set(['spatial-workspace','spatial-endpoint-workspace','rolling-summary','traffic-timeline','offline-replay']);
const names = {overview:'Network overview',flows:'Flow explorer',xray:'Network X-Ray',evidence:'Capture evidence',models:'Model workspace',alerts:'Alerts',simulation:'Simulation lab','active-defense':'Attack control',diagnostics:'Runtime diagnostics'};
const subtitle = document.querySelector('.header-subtitle');
const originalSubtitle = subtitle.textContent;

function element(name, attrs = {}) {
  const node = document.createElementNS(svgNS, name);
  for (const [key,value] of Object.entries(attrs)) node.setAttribute(key,String(value));
  return node;
}
function renderFlat() {
  const svg = $('spatial-flat'); svg.replaceChildren();
  // The static reference silhouette is decoration, never an inferred endpoint.
  svg.append(element('path',{d:'M 320 425 L 500 110 L 680 425 Z M 433 350 L 567 350 L 500 235 Z',fill:'#6884b4','fill-rule':'evenodd',stroke:'#c1d4f5','stroke-width':2,'stroke-linejoin':'bevel'}));
}
function setMode(value) {
  flat = value || unavailable;
  stage.dataset.mode = flat ? 'flat' : '3d';
  $('spatial-canvas').hidden = flat;
  $('spatial-flat').hidden = !flat;
  $('spatial-mode-toggle').setAttribute('aria-pressed',String(flat));
  $('spatial-mode-toggle').textContent = flat ? '3D view' : 'Flat view';
  $('spatial-mode-toggle').disabled = unavailable;
  document.querySelectorAll('[data-view],[data-rotate],[data-zoom]').forEach(button => button.disabled=flat);
  $('spatial-instructions').textContent = flat ? 'Select an endpoint from the list to inspect it' : 'Drag to orbit · use + / − to zoom';
  $('spatial-renderer-status').textContent = unavailable ? 'Flat · 3D acceleration unavailable' : flat ? 'Flat · low power' : '3D · interactive';
  scene?.setEnabled(introOpen && !flat && page==='overview');
  materialControls?.setEnabled(!flat);
  if(flat) renderFlat();
}
function select(next, {focus=false} = {}) {
  const node=endpointSummary(graph,next);
  address=node?.address ?? null;
  const inspector=document.querySelector('.spatial-inspector');
  inspector.hidden=!node;
  stage.classList.toggle('has-selection',!!node);
  for(const button of $('spatial-endpoints').querySelectorAll('button')) button.setAttribute('aria-pressed',String(button.dataset.address===address));
  if(node) {
    $('spatial-address').textContent=node.address;
    $('spatial-node-flows').textContent=node.flows.toLocaleString();
    $('spatial-node-packets').textContent=node.packets.toLocaleString();
    $('spatial-node-bytes').textContent=node.bytes.toLocaleString();
    if(focus)$('spatial-address').focus({preventScroll:true});
  }
  if(flat)renderFlat();
}
function update(nextFlows) {
  flows=Array.isArray(nextFlows)?nextFlows:[];
  graph=observedGraph(flows,$('spatial-protocol').value);
  const list=$('spatial-endpoints');list.replaceChildren();
  if(!graph.nodes.length) {
    const empty=document.createElement('p');empty.textContent='No endpoints available for this protocol.';list.append(empty);
  }
  for(const node of graph.nodes) {
    const button=document.createElement('button');button.type='button';button.dataset.address=node.address;
    const name=document.createElement('strong');name.textContent=node.address;
    const meta=document.createElement('span');meta.textContent=`${node.flows} flows · ${node.packets.toLocaleString()} packets`;
    button.append(name,meta);button.addEventListener('click',()=>select(node.address,{focus:true}));list.append(button);
  }
  $('spatial-list-count').textContent=`(${graph.nodes.length})`;
  $('spatial-empty').hidden=false;
  $('spatial-empty').textContent='Local monitoring. A longer memory. Evidence you can inspect.';
  $('spatial-count').textContent=graph.totalFlows?`${graph.nodes.length} / ${graph.totalNodes} endpoints in index · ${graph.shownFlows} / ${graph.totalFlows} flows indexed · highest byte volume first`:'Waiting for observed flows';
  select(address);
}
function setPage(target) {
  if(!names[target])return;
  page=target;
  if(target!=='overview'&&introOpen)closeIntroduction(false);
  document.body.classList.toggle('on-landing',introOpen&&target==='overview');
  document.querySelectorAll('main > section').forEach(section=> {
    const show=section.id==='spatial-workspace'?introOpen&&target==='overview':target==='overview'?(overview.has(section.id)||section.classList.contains('metrics')||section.classList.contains('visual-grid')):section.id===target;
    section.classList.toggle('workspace-hidden',!show);
  });
  document.querySelector('#overview h1').textContent=names[target];
  subtitle.textContent=target==='overview'?originalSubtitle:'Local tools / '+names[target];
  scene?.setEnabled(introOpen&&target==='overview'&&!flat);
}
function closeIntroduction(focus=true) {
  introOpen=false;story?.destroy();story=null;
  setupStage.stop();
  workspace.hidden=true;workspace.inert=true;workspace.classList.remove('scroll-story');
  setup.hidden=true;setup.setAttribute('aria-hidden','true');
  readinessObserver.disconnect();
  document.body.classList.remove('journey-active');
  document.documentElement.classList.remove('journey-pending');
  document.querySelector('.sidebar').inert=false;
  [...document.querySelector('main').children].forEach(child=>{if(child!==workspace)child.inert=false;});
  setPage(page);
  if(focus){$('overview').scrollIntoView({block:'start'});document.querySelector('#overview h1').focus({preventScroll:true});}
}
async function openIntroduction() {
  if(disposed)return;
  if(introOpen)return;
  window.dispatchEvent(new CustomEvent('argus:intro-navigate'));
  introOpen=true;workspace.hidden=false;workspace.inert=false;setPage('overview');
  setup.hidden=false;setup.setAttribute('aria-hidden','false');
  document.body.classList.add('journey-active');
  document.documentElement.classList.remove('journey-pending');
  document.querySelector('.sidebar').inert=true;
  [...document.querySelector('main').children].forEach(child=>{if(child!==workspace)child.inert=true;});
  story?.destroy();story=createScrollStory(visualStory,()=>scene?.invalidate(),{
    onModeChange:enabled=>queueMicrotask(()=>{if(introOpen)setupStage.configure(enabled,story);}),
  });
  readinessObserver.observe(document.querySelector('[data-onboarding-step="1"]'));
  workspace.scrollIntoView({block:'start'});$('spatial-title').focus({preventScroll:true});
  if(!scenePromise)scenePromise=initializeScene();
  await scenePromise;
  if(!disposed)setMode(flat);
}
window.ArgusWorkspace=Object.freeze({setFlows:update,setPage,closeIntroduction,goToStep:step=>setupStage.goToStep(step),get introOpen(){return introOpen;}});
window.addEventListener('argus:journey-open',()=>void openIntroduction());
$('replay-introduction-button').addEventListener('click',()=>void window.ArgusSetup.open({force:true}));
$('spatial-mode-toggle').addEventListener('click',()=>setMode(!flat));
$('spatial-protocol').addEventListener('change',()=>update(flows));
stage.addEventListener('keydown',event=> {
  if(event.key==='Escape'&&address){select(null);$('spatial-protocol').focus();}
});
$('spatial-deselect').addEventListener('click',()=> {
  const previous=address;select(null);
  const button=[...$('spatial-endpoints').querySelectorAll('button')].find(item=>item.dataset.address===previous);
  if(button && button.checkVisibility())button.focus();else $('spatial-protocol').focus();
});
$('spatial-inspect').addEventListener('click',()=> {
  if(address)window.dispatchEvent(new CustomEvent('argus:spatial-inspect',{detail:{address}}));
});
document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=> {
  scene?.view(button.dataset.view);
  document.querySelectorAll('[data-view]').forEach(item=>item.setAttribute('aria-pressed',String(item===button)));
}));
function clearPreset() { document.querySelectorAll('[data-view]').forEach(button=>button.setAttribute('aria-pressed','false')); }
document.querySelectorAll('[data-zoom]').forEach(button=>button.addEventListener('click',()=>{scene?.zoom(Number(button.dataset.zoom));clearPreset();}));
document.querySelectorAll('[data-rotate]').forEach(button=>button.addEventListener('click',()=>{scene?.rotate(Number(button.dataset.rotate));clearPreset();}));
$('spatial-canvas').addEventListener('pointermove',event=>{if(event.buttons)clearPreset();});
document.querySelectorAll('[data-workspace-jump]').forEach(button=>button.addEventListener('click',()=>{
  if(button.classList.contains('spatial-enter'))setupStage.next();else setupStage.goToStep(0);
}));
const readinessObserver=new IntersectionObserver(entries=>{
  if(introOpen&&entries.some(entry=>entry.isIntersecting)){
    window.dispatchEvent(new CustomEvent('argus:setup-check'));readinessObserver.disconnect();
  }
});
readinessObserver.observe(document.querySelector('[data-onboarding-step="1"]'));
window.addEventListener('pagehide',()=>{disposed=true;setupStage.stop();readinessObserver.disconnect();materialControls?.dispose();story?.destroy();scene?.destroy();},{once:true});
setPage('overview');
update([]);
window.dispatchEvent(new CustomEvent('argus:spatial-ready'));
if(window.ArgusSetup?.pending)void openIntroduction();
async function initializeScene() { try {
  const {createScene}=await import('./scene.mjs?v=room-orbit-1');
  if(disposed)return;
  scene=await createScene($('spatial-canvas'),status=> {
    stage.dataset.renderer=status;
    if(status==='lost'||status==='failed'){unavailable=true;setMode(true);}
  });
  if(disposed){scene.destroy();return;}
  materialControls=bindHeroRoughness($('hero-roughness'),$('hero-roughness-value'),value=>scene.setRoughness(value),storage);
  scene.setStoryReader(()=>story?.progress()??0);setMode(flat);
} catch(error) {
  unavailable=true;stage.dataset.renderer='unavailable';setMode(true);
  console.info('Argus spatial view: using flat fallback.',error.message);
} }
