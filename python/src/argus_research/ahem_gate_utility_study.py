"""Final bounded study today: direct decision supervision versus forecast loss."""
import copy
import json
import tempfile
import time
from pathlib import Path
import torch
from .ahem_bridge_events import AHEM_NAME,EXPECTED_CHECKPOINT_SHA256
from .ahem_forecast_audit import forecast_quality
from .ahem_gate_usefulness_audit import auc_pairwise
from .ahem_gate_utility_targets import utility_targets,utility_loss
from .ahem_outcome_controls import select_blend
from .ahem_outcome_gate import ObservedPriorGate
from .ahem_outcome_gate_study import project_gate
from .ahem_outcome_joint import eligible
from .ahem_phase_experiment import prepare_cache
from .ahem_phase_local_study import combine,cleanup_owned_cache
from .ahem_phase_reader import AhemPhaseReader
from .ahem_population_audit import sha256
from .ahem_reader_experiment import atomic_json,atomic_save
from .flow_v6_final_evaluate import _load_frozen_checkpoint
from .flow_v6_objective import dynamic_loss


def checked_hash(path):
    # Read-only retry for a transient Windows sharing violation. No scientific
    # restart or file overwrite is performed here.
    for attempt in range(4):
        try:
            return sha256(path)
        except PermissionError:
            if attempt==3:
                raise
            time.sleep(.25)


@torch.no_grad()
def evaluate(gate,data,plan):
    device=next(gate.parameters()).device
    features,current,raw,available,target=(data[n].to(device) for n in ('features','current','raw','available','target'))
    prediction=gate(features,current,raw,available)
    logits=gate.network(features)
    coefficient,regret,active=utility_targets(current,raw,target,available,minimum_span=plan['minimum_log_span'])
    trust=logits.sigmoid()
    oracle=current+coefficient*(raw-current)
    benefit=torch.nn.functional.smooth_l1_loss(current,target,reduction='none')-torch.nn.functional.smooth_l1_loss(oracle,target,reduction='none')
    usefulness={}
    for field,name in enumerate(('size','delay')):
        mask=active[:,field]
        labels=benefit[:,field]>1e-8
        usefulness[name]={'active':int(mask.sum()),'useful':int((labels&mask).sum()),
            'trust_auc':auc_pairwise(labels[mask].cpu().numpy(),trust[mask,field].cpu().numpy()),
            'coefficient_mae':float((trust[mask,field]-coefficient[mask,field]).abs().mean()) if mask.any() else None}
    return {'forecast':forecast_quality(prediction.cpu().numpy(),target.cpu().numpy()),
        'usefulness':usefulness,'utility_loss':float(utility_loss(logits,coefficient,regret))}


def fit(gate,data,ewma,plan,seed,variant,output,binding):
    output.mkdir(parents=True)
    val=data['validation']
    chosen=select_blend(val,tuple(plan['blend_grid']))
    controls={'persistence':forecast_quality(val['current'].numpy(),val['target'].numpy()),
        'ewma':forecast_quality(ewma.numpy(),val['target'].numpy()),'fixed_blend':chosen['validation']}
    reference=min(controls.values(),key=lambda m:m['weighted_smooth_l1'])
    initial=evaluate(gate,val,plan)
    atomic_json(binding|{'seed':seed,'variant':variant,'controls':controls,'initial_validation':initial,
        'initial_training':evaluate(gate,data['train'],plan)},output/'RUN.json')
    optimizer=torch.optim.AdamW(gate.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay'])
    generator=torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    train={n:t.to('cuda') for n,t in data['train'].items()}
    teacher,regret,_=utility_targets(train['current'],train['raw'],train['target'],train['available'],minimum_span=plan['minimum_log_span'])
    best,selected=float('inf'),None
    for epoch in range(1,plan['epochs']+1):
        gate.train()
        order=torch.randperm(len(train['target']),generator=generator)
        for start in range(0,len(order),plan['batch']):
            index=order[start:start+plan['batch']].to('cuda')
            if variant=='forecast_only':
                prediction=gate(*(train[n][index] for n in ('features','current','raw','available')))
                loss=dynamic_loss(prediction,train['target'][index])
            else:
                loss=utility_loss(gate.network(train['features'][index]),teacher[index],regret[index])
            if not torch.isfinite(loss):
                raise ValueError('nonfinite objective')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
        gate.eval()
        validation,training=evaluate(gate,val,plan),evaluate(gate,data['train'],plan)
        accepted=eligible(validation,reference,initial,best)
        if accepted:
            best,selected=validation['forecast']['weighted_smooth_l1'],epoch
        state={'schema':'argus.gate-utility-checkpoint/v1','seed':seed,'variant':variant,'epoch':epoch,
            'model_state':gate.state_dict(),'optimizer_state':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),
            'sampler_rng':generator.get_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
            'binding':binding,'validation':validation,'training':training}
        path=output/f'epoch-{epoch:03d}.pt'
        atomic_save(state,path)
        atomic_save(state,output/'latest.pt')
        if accepted:
            atomic_save(state,output/'best.pt')
        row={'epoch':epoch,'selected':accepted,'validation':validation,'training':training,'sha256':checked_hash(path)}
        with (output/'progress.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps({'seed':seed,'variant':variant,'epoch':epoch,'selected':accepted,
            'loss':validation['forecast']['weighted_smooth_l1'],'bytes':validation['forecast']['size_mae_bytes'],
            'size_trust_auc':validation['usefulness']['size']['trust_auc']}),flush=True)
    for path in output.glob('epoch-*.pt'):
        state=torch.load(path,weights_only=True,map_location='cpu')
        ObservedPriorGate().load_state_dict(state['model_state'],strict=True)
        if any(not torch.isfinite(t).all() for t in state['model_state'].values()):
            raise ValueError('invalid checkpoint')
    result={'selected_epoch':selected,'baseline_retained':selected is None,
        'best_sha256':checked_hash(output/'best.pt') if selected else None,'runtime_admission':False}
    atomic_json(result,output/'RESULT.json')
    return result


def run():
    dev=Path('../argus-dev')
    output=dev/'research-runs/ahem-gate-utility-20261003/round-01'
    if output.exists():
        raise FileExistsError('immutable study already exists')
    plan_path=Path('research/ahem-gate-utility-local-20261003.json')
    plan=json.loads(plan_path.read_text())
    if plan['variants']!=['forecast_only','direct_utility'] or plan['epochs']!=12 or plan['early_stopping'] is not False or any(plan[n] for n in ('assessment_access','downloads','runtime_admission')):
        raise ValueError('unexpected last-study protocol')
    source_plan_path=Path('research/ahem-phase-local-study-20261002.json')
    source_plan=json.loads(source_plan_path.read_text())
    prior=dev/'research-runs/ahem-phase-local-study-20261002/round-01'
    recorded=json.loads((prior/'STUDY.json').read_text())['source_sha256']
    sources={role:[dev/'learning'/n for n in source_plan['sources'][role]] for role in ('train','validation')}
    hashes={p.name:checked_hash(p) for ps in sources.values() for p in ps}
    if any(recorded[n]!=h for n,h in hashes.items()):
        raise ValueError('source changed')
    paths={str(s):{'reader':prior/f'trials/{s}-phase-soft/best.pt','gate':dev/f'research-runs/ahem-observed-prior-gate-20261002/round-01/{s}/best.pt'} for s in plan['seeds']}
    parents={s:{n:checked_hash(p) for n,p in ps.items()} for s,ps in paths.items()}
    output.mkdir(parents=True)
    temporary=Path(tempfile.mkdtemp(prefix='argus-gate-utility-'))
    binding={'plan':plan,'plan_sha256':checked_hash(plan_path),'source_plan_sha256':checked_hash(source_plan_path),
        'source_sha256':hashes,'parent_sha256':parents,'temporary_directory':str(temporary),
        'assessment_opened':False,'torch_version':str(torch.__version__),
        'code_sha256':{p.name:checked_hash(p) for p in Path(__file__).parent.glob('*.py')}}
    atomic_json(binding,output/'STUDY.json')
    owned,results={},{}
    for seed in plan['seeds']:
        base,_,_=_load_frozen_checkpoint(dev/'checkpoints/ahem-v6-1-development-20260906/extracted/ahem-v6-1-development-20260906',seed,AHEM_NAME,EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME])
        base=base.to('cuda').eval()
        reader=AhemPhaseReader(base,phase_width=32).to('cuda').eval()
        reader.load_state_dict(torch.load(paths[str(seed)]['reader'],weights_only=True,map_location='cuda')['model_state'],strict=True)
        reader.requires_grad_(False)
        data,ewma={},None
        for role,ps in sources.items():
            caches=[]
            for i,source in enumerate(ps):
                path=temporary/f'{seed}-{role}-{i}.pt'
                with torch.no_grad():
                    caches.append(prepare_cache(base,source,role,seed,source_plan,binding['source_plan_sha256'],path,official=False,samples=plan['samples_per_source'][role]))
                owned[path.name]=checked_hash(path)
                atomic_json(json.loads(path.with_suffix('.manifest.json').read_text()),output/f'cache-{path.stem}.manifest.json')
            tensors=combine(caches,role)['tensors']
            data[role]=project_gate(reader,tensors,plan['batch'],'cuda')
            if role=='validation':
                ewma=tensors['ewma']
        gate=ObservedPriorGate().to('cuda')
        gate.load_state_dict(torch.load(paths[str(seed)]['gate'],weights_only=True,map_location='cuda')['model_state'],strict=True)
        if sum(p.numel() for p in gate.parameters())!=plan['trainable_parameters']:
            raise ValueError('parameter contract changed')
        for variant in plan['variants']:
            results[f'{seed}/{variant}']=fit(copy.deepcopy(gate),data,ewma,plan,seed,variant,output/f'{seed}-{variant}',binding)
        del data,tensors,caches,base,reader,gate
        torch.cuda.empty_cache()
    if any(checked_hash(p)!=hashes[p.name] for ps in sources.values() for p in ps) or any(checked_hash(p)!=parents[s][n] for s,ps in paths.items() for n,p in ps.items()):
        raise ValueError('bound input changed')
    atomic_json({'selections':results,'assessment_opened':False,'runtime_admission':False},output/'FROZEN.json')
    removed=cleanup_owned_cache(temporary,owned)
    # Exact owned sidecars only; provenance copies already exist in output.
    for name in removed:
        temporary.joinpath(name).with_suffix('.manifest.json').unlink()
    temporary.rmdir()
    atomic_json({'removed':removed,'sidecars_preserved_in_output':True,'temporary_directory_removed':True},output/'CLEANUP.json')
    print(json.dumps({'stage':'complete','results':results}),flush=True)


if __name__=='__main__':
    torch.set_num_threads(4)
    run()
