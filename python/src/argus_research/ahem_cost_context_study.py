"""Bounded one-seed paired continuous-cost readout experiment; research only."""
import json
import tempfile
from pathlib import Path
import torch
from .ahem_cost_context_readout import CostContextReadout
from .ahem_forecast_audit import forecast_quality
from .ahem_information_probe import labels,auc_pairwise
from .ahem_outcome_gate import ObservedPriorGate
from .flow_v6_objective import dynamic_loss


@torch.no_grad()
def evaluate(model,data,plan):
    device=next(model.parameters()).device
    f,c,p,r,a=(data[n].to(device) for n in ('features','context','current','raw','available'))
    prediction=model(f,c,p,r,a).cpu()
    trust=model.trust(f,c,a).cpu()
    y,m=labels(data,{'minimum_log_span':1e-4,'benefit_epsilon':1e-8})
    fields={name:{'active':int(m[:,i].sum()),'useful':int((m[:,i]&y[:,i]).sum()),
        'trust_auc':auc_pairwise(y[m[:,i],i].numpy(),trust[m[:,i],i].numpy()),
        'mean_trust':float(trust[:,i].mean())} for i,name in enumerate(('size','delay'))}
    values=torch.nn.functional.smooth_l1_loss(prediction,data['target'],reduction='none')
    # Preserve lightweight loss/trust evidence, not feature tensors or payloads.
    return {'forecast':forecast_quality(prediction.numpy(),data['target'].numpy()),'usefulness':fields,
        'loss_and_trust':{'field_huber':values.tolist(),'trust':trust.tolist()}}


def passes_source(metrics,reference,initial,best):
    m=metrics['forecast']
    return (m['weighted_smooth_l1']<min(reference['weighted_smooth_l1'],initial['forecast']['weighted_smooth_l1'],best)
        and all(m[n]<=reference[n] for n in ('size_smooth_l1','delay_smooth_l1','size_mae_bytes')))


def fit(model,train,validation,controls,initial,plan,variant,output,binding,matched=None):
    from .ahem_physical_objective_probe import byte_objective
    from .ahem_reader_experiment import atomic_json,atomic_save
    from .ahem_gate_utility_study import checked_hash
    output.mkdir()
    device='cuda'
    model=model.to(device)
    training={n:t.to(device) for n,t in train.items()}
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=plan['learning_rate'],weight_decay=plan['weight_decay'])
    generator=torch.Generator().manual_seed(plan['seed'])
    initial_state={n:t.detach().cpu().clone() for n,t in model.state_dict().items()}
    atomic_json(binding|{'variant':variant,'contract':model.research_contract(),'initial_validation':initial,
        'controls':controls,'initial_training':evaluate(model,train,plan)},output/'RUN.json')
    atomic_save(initial_state,output/'INITIAL.pt')
    rows=[]
    selected=None
    best={k:float('inf') for k in validation}
    for epoch in range(1,plan['epochs']+1):
        model.train()
        order=torch.randperm(len(training['target']),generator=generator)
        for start in range(0,len(order),plan['batch']):
            i=order[start:start+plan['batch']].to(device)
            prediction=model(*(training[n][i] for n in ('features','context','current','raw','available')))
            loss=dynamic_loss(prediction,training['target'][i])+plan['physical_weight']*byte_objective(prediction,training['target'][i])
            if not torch.isfinite(loss):
                raise ValueError('nonfinite cost objective')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters,plan['gradient_clip'],error_if_nonfinite=True)
            optimizer.step()
        model.eval()
        results={k:evaluate(model,d,plan) for k,d in validation.items()}
        checks={k:passes_source(v,controls[k]['reference'],initial[k],best[k]) for k,v in results.items()}
        paired={k:True if matched is None else v['forecast']['weighted_smooth_l1']<matched[epoch-1]['validation'][k]['forecast']['weighted_smooth_l1'] for k,v in results.items()}
        accepted=all(checks.values()) and all(paired.values())
        if accepted:
            selected=epoch
            best={k:v['forecast']['weighted_smooth_l1'] for k,v in results.items()}
        training_metrics=evaluate(model,train,plan)
        state={'schema':'argus.cost-context-checkpoint/v1','variant':variant,'epoch':epoch,'model_state':model.state_dict(),
            'optimizer_state':optimizer.state_dict(),'sampler_rng':generator.get_state(),'binding':binding,
            'validation':results,'training':training_metrics,'selected':accepted,'source_checks':checks,'paired_checks':paired}
        path=output/f'epoch-{epoch:03d}.pt'
        atomic_save(state,path)
        atomic_save(state,output/'latest.pt')
        if accepted:
            atomic_save(state,output/'best.pt')
        row={'epoch':epoch,'selected':accepted,'validation':results,'training':training_metrics,
            'source_checks':checks,'paired_checks':paired,'sha256':checked_hash(path)}
        rows.append(row)
        with (output/'progress.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps({'variant':variant,'epoch':epoch,'selected':accepted,
            'sources':{k:{'loss':v['forecast']['weighted_smooth_l1'],'bytes':v['forecast']['size_mae_bytes']} for k,v in results.items()}}),flush=True)
    result={'selected_epoch':selected,'baseline_retained':selected is None,'best_sha256':checked_hash(output/'best.pt') if selected else None,
        'runtime_admission':False,'final_validation':rows[-1]['validation']}
    atomic_json(result,output/'RESULT.json')
    return rows,result


def run():
    from .ahem_bridge_events import AHEM_NAME,EXPECTED_CHECKPOINT_SHA256
    from .ahem_gate_utility_study import checked_hash
    from .ahem_outcome_controls import select_blend
    from .ahem_outcome_gate_study import project_gate
    from .ahem_phase_experiment import prepare_cache
    from .ahem_phase_local_study import cleanup_owned_cache
    from .ahem_phase_reader import AhemPhaseReader
    from .ahem_reader_experiment import atomic_json,atomic_save
    from .flow_v6_final_evaluate import _load_frozen_checkpoint
    dev=Path('../argus-dev')
    output=dev/'research-runs/ahem-cost-context-20261004/round-01'
    if output.exists():
        raise FileExistsError('immutable study exists')
    plan_path=Path('research/ahem-cost-context-local-20261004.json')
    plan=json.loads(plan_path.read_text())
    if (plan['variants']!=['summary_cost','context_cost'] or plan['epochs']!=12 or plan['seed']!=20260906
        or plan['physical_normalization_bytes']!=1500 or plan['early_stopping'] is not False
        or any(plan[n] for n in ('assessment_access','downloads','runtime_admission','reader_training','retrieval_training','automatic_replication'))):
        raise ValueError('unexpected experiment scope')
    source_plan_path=Path('research/ahem-phase-local-study-20261002.json')
    source_plan=json.loads(source_plan_path.read_text())
    prior=dev/'research-runs/ahem-phase-local-study-20261002/round-01'
    recorded=json.loads((prior/'STUDY.json').read_text())['source_sha256']
    sources={role:[dev/'learning'/n for n in source_plan['sources'][role]] for role in ('train','validation')}
    hashes={p.name:checked_hash(p) for ps in sources.values() for p in ps}
    if any(recorded[n]!=h for n,h in hashes.items()) or len(set(hashes.values()))!=5:
        raise ValueError('source hash or duplicate mismatch')
    if {p.name[:10] for p in sources['train']}&{p.name[:10] for p in sources['validation']}:
        raise ValueError('source dates overlap')
    seed=plan['seed']
    reader_path=prior/f'trials/{seed}-phase-soft/best.pt'
    gate_path=dev/f'research-runs/ahem-observed-prior-gate-20261002/round-01/{seed}/best.pt'
    parents={str(p):checked_hash(p) for p in (reader_path,gate_path)}
    output.mkdir(parents=True)
    tmp=Path(tempfile.mkdtemp(prefix='argus-cost-context-'))
    binding={'plan':plan,'plan_sha256':checked_hash(plan_path),'source_plan_sha256':checked_hash(source_plan_path),
        'source_sha256':hashes,'parents':parents,'temporary':str(tmp),'assessment_opened':False,
        'torch_version':str(torch.__version__),'code_sha256':{p.name:checked_hash(p) for p in Path(__file__).parent.glob('*.py')}}
    atomic_json(binding,output/'STUDY.json')
    base,_,_=_load_frozen_checkpoint(dev/'checkpoints/ahem-v6-1-development-20260906/extracted/ahem-v6-1-development-20260906',seed,AHEM_NAME,EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME])
    base=base.to('cuda').eval()
    reader=AhemPhaseReader(base,phase_width=32).to('cuda').eval()
    reader.load_state_dict(torch.load(reader_path,weights_only=True,map_location='cuda')['model_state'],strict=True)
    reader.requires_grad_(False)
    owned,data={},{}
    for role,paths in sources.items():
        for i,source in enumerate(paths):
            path=tmp/f'{role}-{i}.pt'
            with torch.no_grad():
                cache=prepare_cache(base,source,role,seed,source_plan,binding['source_plan_sha256'],path,official=False,samples=plan['samples_per_source'][role])
                t=cache['tensors']
                d=project_gate(reader,t,plan['batch'],'cuda')
                d['context']=t['query_phase']
                d['ewma']=t['ewma']
            owned[path.name]=checked_hash(path)
            atomic_json(json.loads(path.with_suffix('.manifest.json').read_text()),output/f'{role}-{i}.manifest.json')
            atomic_json({'source':source.name,'coordinates':t['coordinates'].tolist()},output/f'{role}-{i}-coordinates.json')
            data[f'{role}-{i}']=d
            print(json.dumps({'stage':'projected','source':source.name,'samples':len(d['target'])}),flush=True)
            del cache,t
    train={n:torch.cat([d[n] for k,d in data.items() if k.startswith('train-')]) for n in data['train-0']}
    validation={k:d for k,d in data.items() if k.startswith('validation-')}
    context=train['context']
    if context.shape[1]!=plan['context_dim']:
        raise ValueError('causal context dimension changed')
    mean,scale=context.mean(0),context.std(0,unbiased=False).clamp_min(1e-6)
    # Reusable compact snapshot; targets remain outside every model input path.
    snapshot=tmp/'features.pt'
    atomic_save(data,snapshot)
    owned[snapshot.name]=checked_hash(snapshot)
    atomic_json({'sha256':owned[snapshot.name],'mean_fit_on':'training only','rows':{k:len(d['target']) for k,d in data.items()}},output/'SNAPSHOT.json')
    parent=ObservedPriorGate()
    parent.load_state_dict(torch.load(gate_path,weights_only=True,map_location='cpu')['model_state'],strict=True)
    controls={}
    for k,d in validation.items():
        blend=select_blend(d,tuple(plan['blend_grid']))
        metrics={'persistence':forecast_quality(d['current'].numpy(),d['target'].numpy()),
            'ewma':forecast_quality(d['ewma'].numpy(),d['target'].numpy()),'fixed_blend':blend['validation']}
        controls[k]={'metrics':metrics,'reference':min(metrics.values(),key=lambda m:m['weighted_smooth_l1']),
            'blend_coefficients':blend['coefficients']}
    rows,results={},{}
    for variant in plan['variants']:
        model=CostContextReadout(parent,mean,scale,context_enabled=variant=='context_cost')
        initial={k:evaluate(model,d,plan) for k,d in validation.items()}
        rows[variant],results[variant]=fit(model,train,validation,controls,initial,plan,variant,output/variant,binding,
            matched=rows.get('summary_cost') if variant=='context_cost' else None)
    # Strict artifact validation does not alter model selection.
    validated=[]
    for variant in plan['variants']:
        for path in sorted((output/variant).glob('epoch-*.pt')):
            state=torch.load(path,weights_only=True,map_location='cpu')
            restored=CostContextReadout(parent,mean,scale,context_enabled=variant=='context_cost')
            restored.load_state_dict(state['model_state'],strict=True)
            if any(not torch.isfinite(t).all() for t in state['model_state'].values()):
                raise ValueError('nonfinite checkpoint')
            if not torch.equal(state['model_state']['context_mean'],mean) or not torch.equal(state['model_state']['context_scale'],scale):
                raise ValueError('training statistics mutated')
            validated.append({'file':str(path.relative_to(output)),'sha256':checked_hash(path)})
    if any(checked_hash(p)!=hashes[p.name] for ps in sources.values() for p in ps) or any(checked_hash(Path(p))!=h for p,h in parents.items()):
        raise ValueError('bound input changed')
    atomic_json({'results':results,'validated_checkpoints':validated,'runtime_admission':False,
        'assessment_opened':False,'evidence':plan['evidence']},output/'FROZEN.json')
    removed=cleanup_owned_cache(tmp,owned)
    for n in removed:
        sidecar=tmp.joinpath(n).with_suffix('.manifest.json')
        if sidecar.exists():
            sidecar.unlink()
    tmp.rmdir()
    atomic_json({'removed':removed,'temporary_directory_removed':True},output/'CLEANUP.json')
    print(json.dumps({'stage':'complete','selections':{v:r['selected_epoch'] for v,r in results.items()}}),flush=True)


if __name__=='__main__':
    torch.set_num_threads(4)
    run()
