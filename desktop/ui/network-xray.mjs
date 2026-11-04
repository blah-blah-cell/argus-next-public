// The index consumes only the same filtered topology used by the canvas.
export function endpointRows(nodes,query='',limit=200) {
  const term=String(query).trim().toLowerCase(),seen=new Set();
  const rows=[];
  for(const node of Array.isArray(nodes)?nodes:[]) {
    if(!node||typeof node.id!=='string'||!node.id.trim()||seen.has(node.id))continue;
    seen.add(node.id);
    const label=String(node.entityName||node.hostname||node.ip||node.id);
    const address=String(node.ip||node.id),role=String(node.role||'Endpoint');
    if(![node.id,label,address,role,node.vendor].some(value=>String(value??'').toLowerCase().includes(term)))continue;
    const count=value=>Number.isFinite(value)&&value>0?value:0;
    rows.push({id:node.id,label,address,role,bytes:count(node.totalInBytes??node.total_in_bytes)+count(node.totalOutBytes??node.total_out_bytes)});
  }
  rows.sort((a,b)=>b.bytes-a.bytes||a.id.localeCompare(b.id));
  return {total:rows.length,rows:rows.slice(0,Math.max(0,Math.min(200,Math.floor(Number(limit)||0))))};
}
export function mountEndpointIndex(root,host) {
  const search=root.getElementById('xray-endpoint-search'),list=root.getElementById('xray-endpoint-list');
  if(!search||!list)return;
  const count=root.getElementById('xray-index-count'),empty=root.getElementById('xray-index-empty');
  let nodes=[],loaded=false,selected=null,signature='';
  const select=()=>list.querySelectorAll('button').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.endpointId===selected)));
  function render() {
    const result=endpointRows(nodes,search.value);
    const next=JSON.stringify(result);
    if(next!==signature) {
      const focused=root.activeElement?.dataset?.endpointId;
      list.replaceChildren();signature=next;
      for(const row of result.rows) {
        const item=root.createElement('li'),button=root.createElement('button');
        button.type='button';button.dataset.endpointId=row.id;button.setAttribute('aria-controls','xray-node-drawer');
        const title=root.createElement('strong'),meta=root.createElement('span');
        title.textContent=row.label;
        meta.textContent=`${row.address} · ${row.role} · ${row.bytes.toLocaleString()} B`;
        button.append(title,meta);item.append(button);list.append(item);
        button.addEventListener('click',()=>host.dispatchEvent(new CustomEvent('argus:xray-select',{detail:{id:row.id}})));
        if(focused===row.id)button.focus({preventScroll:true});
      }
    }
    count.textContent=loaded?`${result.rows.length} of ${result.total} matching endpoints · highest volume first`:'No topology loaded.';
    empty.hidden=result.total>0;
    empty.textContent=!loaded?'Refresh to load observed endpoints. Capture traffic first if none are available.':search.value.trim()?'No endpoints match this search. Clear it or choose All.':'No endpoints in this filter. Choose All, or capture traffic if the network is empty.';
    select();
  }
  search.addEventListener('input',render);
  host.addEventListener('argus:xray-data',event=>{nodes=event.detail?.nodes;loaded=true;render();});
  host.addEventListener('argus:xray-selection',event=>{selected=event.detail?.id??null;select();});
  render();host.dispatchEvent(new CustomEvent('argus:xray-ready'));
}
if(typeof document!=='undefined')mountEndpointIndex(document,window);
