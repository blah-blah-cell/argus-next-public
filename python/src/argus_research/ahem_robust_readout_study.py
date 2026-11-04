"""Fixed six-trial robust readout study, with frozen reader and backbone."""
import copy
import json
from pathlib import Path
import torch
from .ahem_bridge_events import AHEM_NAME, EXPECTED_CHECKPOINT_SHA256
from .ahem_forecast_audit import forecast_quality
from .ahem_outcome_controls import select_blend
from .ahem_outcome_gate import ObservedPriorGate
from .ahem_outcome_gate_study import project_gate
from .ahem_outcome_joint import eligible
from .ahem_phase_experiment import prepare_cache
from .ahem_phase_local_study import combine, cleanup_owned_cache
from .ahem_phase_reader import AhemPhaseReader, phase_infer
from .ahem_physical_objective_probe import byte_objective
from .ahem_prediction_capacity_probe import historical_weighted_median
from .ahem_population_audit import sha256
from .ahem_reader_experiment import atomic_json, atomic_save
from .ahem_robust_readout import RobustObservedReadout
from .flow_v6_final_evaluate import _load_frozen_checkpoint
from .flow_v6_objective import dynamic_loss


@torch.no_grad()
def project(reader, tensors, batch):
    projected = project_gate(reader,tensors,batch,'cuda')
    medians = []
    for start in range(0,len(tensors['target']),batch):
        index = slice(start,start+batch)
        out = phase_infer(reader,tensors,index,'cuda')
        medians.append(historical_weighted_median(out.successor_dynamic,out.retrieval_weights,out.valid,tensors['persistence'][index].to('cuda')).cpu())
    return projected | {'median': torch.cat(medians)}


def predict(model,data,index=slice(None)):
    return model(*(data[n][index].to('cuda') for n in ('features','current','raw','median','available')))


@torch.no_grad()
def evaluate(model,data):
    return {'forecast': forecast_quality(predict(model,data).cpu().numpy(),data['target'].cpu().numpy())}


def fit(model,data,ewma,plan,seed,variant,output,binding):
    output.mkdir(parents=True)
    val = data['validation']
    chosen = select_blend(val)
    controls = {n: forecast_quality(val[field].numpy(),val['target'].numpy()) for n,field in [('persistence','current'),('median','median')]}
    controls['ewma'] = forecast_quality(ewma.numpy(),val['target'].numpy())
    controls['fixed_blend'] = chosen['validation']
    reference = min(controls.values(),key=lambda m:m['weighted_smooth_l1'])
    initial = evaluate(model,val)
    atomic_json(binding | {'seed': seed, 'variant': variant, 'initial_validation': initial, 'controls': controls},output/'RUN.json')
    optimizer = torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay'])
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    best,selected,stale = float('inf'),None,0
    train = {n:t.to('cuda') for n,t in data['train'].items()}
    for epoch in range(1,plan['epochs']+1):
        model.train()
        order = torch.randperm(len(train['target']),generator=generator)
        for start in range(0,len(order),plan['batch']):
            index = order[start:start+plan['batch']].to('cuda')
            prediction = predict(model,train,index)
            loss = dynamic_loss(prediction,train['target'][index]) + plan['physical_weight']*byte_objective(prediction,train['target'][index])
            if not torch.isfinite(loss):
                raise ValueError('nonfinite objective')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
        model.eval()
        validation,training = evaluate(model,val),evaluate(model,data['train'])
        accepted = eligible(validation,reference,initial,best)
        if accepted:
            best,selected,stale = validation['forecast']['weighted_smooth_l1'],epoch,0
        else:
            stale += 1
        state = {'schema':'argus.robust-readout-checkpoint/v1','seed':seed,'variant':variant,'epoch':epoch,
            'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),'validation':validation,
            'training':training,'torch_rng':torch.get_rng_state(),'sampler_rng':generator.get_state(),
            'cuda_rng':torch.cuda.get_rng_state_all(),'binding':binding}
        path = output/f'epoch-{epoch:03d}.pt'
        atomic_save(state,path)
        atomic_save(state,output/'latest.pt')
        if accepted:
            atomic_save(state,output/'best.pt')
        with (output/'progress.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps({'epoch':epoch,'selected':accepted,'validation':validation,'training':training,'sha256':sha256(path)},allow_nan=False)+'\n')
        print(json.dumps({'seed':seed,'variant':variant,'epoch':epoch,'selected':accepted,'loss':validation['forecast']['weighted_smooth_l1'],'bytes':validation['forecast']['size_mae_bytes']}),flush=True)
        if stale >= plan['patience']:
            break
    for path in output.glob('epoch-*.pt'):
        state = torch.load(path,weights_only=True,map_location='cpu')
        restored = RobustObservedReadout(median_enabled=variant=='median_enabled')
        restored.load_state_dict(state['model_state'],strict=True)
        if any(not torch.isfinite(t).all() for t in state['model_state'].values()):
            raise ValueError('invalid checkpoint')
    result = {'selected_epoch':selected,'baseline_retained':selected is None,'best_sha256':sha256(output/'best.pt') if selected else None,'runtime_admission':False}
    atomic_json(result,output/'RESULT.json')
    return result


def run():
    dev = Path('../argus-dev')
    output = dev/'research-runs/ahem-robust-readout-20261003/round-01'
    if output.exists():
        raise FileExistsError('immutable study already exists')
    plan_path = Path('research/ahem-robust-readout-local-20261003.json')
    plan = json.loads(plan_path.read_text())
    if plan['variants'] != ['median_disabled','median_enabled'] or any(plan[n] for n in ('assessment_access','downloads','runtime_admission')):
        raise ValueError('unexpected protocol')
    source_plan_path = Path('research/ahem-phase-local-study-20261002.json')
    source_plan = json.loads(source_plan_path.read_text())
    prior = dev/'research-runs/ahem-phase-local-study-20261002/round-01'
    recorded = json.loads((prior/'STUDY.json').read_text())['source_sha256']
    sources = {role:[dev/'learning'/n for n in source_plan['sources'][role]] for role in ('train','validation')}
    hashes = {p.name:sha256(p) for ps in sources.values() for p in ps}
    if any(recorded[n]!=h for n,h in hashes.items()):
        raise ValueError('source changed')
    paths = {str(s):{'reader':prior/f'trials/{s}-phase-soft/best.pt','gate':dev/f'research-runs/ahem-observed-prior-gate-20261002/round-01/{s}/best.pt'} for s in plan['seeds']}
    parent_hashes = {s:{n:sha256(p) for n,p in ps.items()} for s,ps in paths.items()}
    output.mkdir(parents=True)
    temporary = output/'temporary-projections'
    temporary.mkdir()
    binding = {'plan':plan,'plan_sha256':sha256(plan_path),'source_plan_sha256':sha256(source_plan_path),
        'source_sha256':hashes,'parent_sha256':parent_hashes,'assessment_opened':False,
        'torch_version':str(torch.__version__),'code_sha256':{p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}}
    atomic_json(binding,output/'STUDY.json')
    owned,results = {},{}
    for seed in plan['seeds']:
        base,_,_ = _load_frozen_checkpoint(dev/'checkpoints/ahem-v6-1-development-20260906/extracted/ahem-v6-1-development-20260906',seed,AHEM_NAME,EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME])
        base = base.to('cuda').eval()
        reader = AhemPhaseReader(base,phase_width=32).to('cuda').eval()
        reader.load_state_dict(torch.load(paths[str(seed)]['reader'],weights_only=True,map_location='cuda')['model_state'],strict=True)
        reader.requires_grad_(False)
        data,ewma = {},None
        for role,ps in sources.items():
            caches = []
            for i,source in enumerate(ps):
                path = temporary/f'{seed}-{role}-{i}.pt'
                with torch.no_grad():
                    caches.append(prepare_cache(base,source,role,seed,source_plan,binding['source_plan_sha256'],path,official=False,samples=plan['samples_per_source'][role]))
                owned[path.name] = sha256(path)
            tensors = combine(caches,role)['tensors']
            data[role] = project(reader,tensors,plan['batch'])
            if role=='validation':
                ewma = tensors['ewma']
        parent = ObservedPriorGate().to('cuda')
        parent.load_state_dict(torch.load(paths[str(seed)]['gate'],weights_only=True,map_location='cuda')['model_state'],strict=True)
        for variant in plan['variants']:
            model = RobustObservedReadout(median_enabled=variant=='median_enabled').to('cuda')
            model.initialize_from_gate(parent)
            if sum(p.numel() for p in model.parameters())!=plan['trainable_parameters']:
                raise ValueError('parameter contract mismatch')
            results[f'{seed}/{variant}'] = fit(model,data,ewma,plan,seed,variant,output/f'{seed}-{variant}',binding)
        del data,caches,tensors,base,reader,parent,model
        torch.cuda.empty_cache()
    if any(sha256(p)!=hashes[p.name] for ps in sources.values() for p in ps) or any(sha256(p)!=parent_hashes[s][n] for s,ps in paths.items() for n,p in ps.items()):
        raise ValueError('bound input changed')
    atomic_json({'selections':results,'assessment_opened':False,'runtime_admission':False},output/'FROZEN.json')
    atomic_json({'removed':cleanup_owned_cache(temporary,owned),'manifest_sidecars_preserved':True},output/'CLEANUP.json')
    print(json.dumps({'stage':'complete','results':results}),flush=True)


if __name__=='__main__':
    torch.set_num_threads(4)
    run()
