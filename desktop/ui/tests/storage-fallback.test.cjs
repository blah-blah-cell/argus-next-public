const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'../app.js'),'utf8');
function extract(name){const start=source.indexOf(`function ${name}(`);const end=source.indexOf('\n}',start)+2;return source.slice(start,end);}

test('blocked optional storage does not abort theme boot or lose session selection',()=>{
  const attrs={};
  const context=vm.createContext({sessionTheme:null,localStorage:{getItem(){throw Error('blocked');},setItem(){throw Error('blocked');}},
    window:{matchMedia:()=>({matches:false})},$:()=>null,document:{documentElement:{setAttribute:(key,value)=>attrs[key]=value},querySelectorAll:()=>[]}});
  vm.runInContext(extract('getStoredTheme')+'\n'+extract('setTheme'),context);
  assert.equal(context.getStoredTheme(),'dark');
  context.setTheme(context.getStoredTheme(),false);
  assert.equal(attrs['data-theme'],'dark');
  context.setTheme('light',false);
  assert.equal(context.getStoredTheme(),'light');
});

test('blocked preference writes do not interrupt applying session controls or table updates',()=>{
  const state={uiPreferences:{density:'compact',reduceMotion:true,refreshMs:2000,flowRowLimit:50},flowPagination:{},flows:[{}]};
  const root={dataset:{}};const controls={};let renders=0;
  const context=vm.createContext({state,localStorage:{setItem(){throw Error('quota exceeded');}},$:id=>controls[id]??(controls[id]={}),
    document:{documentElement:root,querySelectorAll:()=>[]},renderTable(){renders++;}});
  vm.runInContext(extract('applyUiPreferences'),context);
  context.applyUiPreferences();
  assert.equal(root.dataset.density,'compact');
  assert.equal(root.dataset.motion,'reduced');
  assert.equal(controls['flow-page-size'].value,'50');
  assert.equal(state.flowPagination.pageSize,50);
  assert.equal(renders,1);
});
