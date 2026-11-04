"""One-seed, train-fit information diagnostic. Not a promoted AHEM model."""
import json
import tempfile
from pathlib import Path
import torch
from torch.nn import functional as F
from .ahem_forecast_audit import forecast_quality
from .ahem_gate_utility_targets import utility_targets


def auc_pairwise(truth, scores):
    # Same tie-half statistic as the existing audit, with no extraction imports.
    positive, negative = scores[truth],scores[~truth]
    if not len(positive) or not len(negative):
        return None
    differences = positive[:,None]-negative[None,:]
    return float(((differences>0)+.5*(differences==0)).mean())


def labels(data, plan):
    coefficient, _, active = utility_targets(data['current'], data['raw'], data['target'],
        data['available'], minimum_span=plan['minimum_log_span'])
    oracle = data['current'] + coefficient * (data['raw'] - data['current'])
    benefit = F.smooth_l1_loss(data['current'], data['target'], reduction='none') - F.smooth_l1_loss(oracle, data['target'], reduction='none')
    return benefit > plan['benefit_epsilon'], active


def choose_threshold(scores, truth):
    if not truth.any() or truth.all():
        raise ValueError('training field lacks both classes')
    candidates = torch.cat((scores.unique(), scores.new_tensor([1.000001])))
    ranked = []
    for threshold in candidates:
        chosen = scores >= threshold
        j = chosen[truth].double().mean() - chosen[~truth].double().mean()
        ranked.append((float(j), float(threshold)))
    return max(ranked)[1]


def best_coefficient(data, field, chosen, grid):
    current, raw, target = (data[n][:, field].double() for n in ('current','raw','target'))
    return min(grid, key=lambda g: (float(F.smooth_l1_loss(current + g * chosen * (raw-current), target)), g))


def fit_probe(x, truth, mask, plan):
    x = x[mask].double()
    y = truth[mask].double()
    if not y.any() or y.all():
        raise ValueError('cannot fit one-class training probe')
    mean, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6)
    z = (x-mean)/scale
    weight = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.tensor(float(torch.logit(y.mean())), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([weight,bias], max_iter=plan['max_iterations'], line_search_fn='strong_wolfe')
    evaluations = []
    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(z @ weight + bias, y) + .5*plan['l2']*weight.square().sum()
        if not torch.isfinite(loss):
            raise ValueError('nonfinite probe loss')
        loss.backward()
        evaluations.append(float(loss.detach()))
        return loss
    opt.step(closure)
    state = {n:t.detach() for n,t in {'mean':mean,'scale':scale,'weight':weight,'bias':bias}.items()}
    if any(not torch.isfinite(t).all() for t in state.values()):
        raise ValueError('nonfinite probe state')
    return state | {'closure_evaluations':len(evaluations),'last_closure_loss':evaluations[-1]}


def score(state, x):
    return (((x.double()-state['mean'])/state['scale']) @ state['weight'] + state['bias']).sigmoid()


def metrics(data, scores, thresholds, coefficients, truth, active):
    prediction = data['current'].double().clone()
    fields = {}
    for f,name in enumerate(('size','delay')):
        chosen = (scores[:,f] >= thresholds[f]) & active[:,f]
        prediction[:,f] += coefficients[f] * chosen * (data['raw'][:,f]-data['current'][:,f])
        m,y,s = active[:,f],truth[:,f],scores[:,f]
        tp,fp,fn = int((chosen&y).sum()),int((chosen&~y).sum()),int((m&y&~chosen).sum())
        fields[name] = {'active':int(m.sum()),'useful':int((m&y).sum()),'not_useful':int((m&~y).sum()),
            'auc':auc_pairwise(y[m].numpy(),s[m].numpy()),'precision':tp/(tp+fp) if tp+fp else None,
            'recall':tp/(tp+fn) if tp+fn else None,'false_trust_count':fp,
            'false_trust_rate':fp/int((m&~y).sum()) if (m&~y).any() else None,
            'threshold':thresholds[f],'coefficient':coefficients[f],
            'introduced_log_harm':float((F.smooth_l1_loss(prediction[:,f],data['target'][:,f].double(),reduction='none')-F.smooth_l1_loss(data['current'][:,f].double(),data['target'][:,f].double(),reduction='none')).clamp_min(0).sum())}
    return {'fields':fields,'forecast':forecast_quality(prediction.numpy(),data['target'].numpy())}


def run():
    # Extraction dependencies are intentionally lazy: unit tests need neither
    # PCAP readers nor the detection bridge's scipy/sklearn dependency tree.
    from .ahem_bridge_events import AHEM_NAME, EXPECTED_CHECKPOINT_SHA256
    from .ahem_gate_utility_study import checked_hash
    from .ahem_outcome_gate import ObservedPriorGate
    from .ahem_outcome_gate_study import project_gate
    from .ahem_phase_experiment import prepare_cache
    from .ahem_phase_local_study import cleanup_owned_cache
    from .ahem_phase_reader import AhemPhaseReader
    from .ahem_reader_experiment import atomic_json, atomic_save
    from .flow_v6_final_evaluate import _load_frozen_checkpoint
    dev = Path('../argus-dev')
    output = dev/'research-runs/ahem-information-probe-20261004/round-01'
    if output.exists():
        raise FileExistsError('immutable diagnostic already exists')
    plan_path = Path('research/ahem-information-probe-20261004.json')
    plan = json.loads(plan_path.read_text())
    if any(plan[n] for n in ('assessment_access','downloads','runtime_admission','validation_selection')) or plan['seed']!=20260906:
        raise ValueError('unexpected diagnostic scope')
    source_path = Path('research/ahem-phase-local-study-20261002.json')
    source_plan = json.loads(source_path.read_text())
    prior = dev/'research-runs/ahem-phase-local-study-20261002/round-01'
    reader_path = prior/f"trials/{plan['seed']}-phase-soft/best.pt"
    gate_path = dev/f"research-runs/ahem-observed-prior-gate-20261002/round-01/{plan['seed']}/best.pt"
    sources = {role:[dev/'learning'/n for n in source_plan['sources'][role]] for role in ('train','validation')}
    dates = {role:{p.name[:10] for p in paths} for role,paths in sources.items()}
    if dates['train'] & dates['validation']:
        raise ValueError('source dates overlap')
    hashes = {p.name:checked_hash(p) for ps in sources.values() for p in ps}
    if len(set(hashes.values())) != sum(map(len,sources.values())):
        raise ValueError('duplicate source bytes')
    recorded = json.loads((prior/'STUDY.json').read_text())['source_sha256']
    if any(recorded[n]!=h for n,h in hashes.items()):
        raise ValueError('source parity failed')
    parents = {str(p):checked_hash(p) for p in (reader_path,gate_path)}
    output.mkdir(parents=True)
    temporary = Path(tempfile.mkdtemp(prefix='argus-information-probe-'))
    binding = {'plan':plan,'plan_sha256':checked_hash(plan_path),'source_plan_sha256':checked_hash(source_path),
        'sources':hashes,'parents':parents,'assessment_opened':False,'temporary':str(temporary),
        'code_sha256':{p.name:checked_hash(p) for p in Path(__file__).parent.glob('*.py')}}
    atomic_json(binding,output/'FROZEN.json')
    seed = plan['seed']
    base,_,_ = _load_frozen_checkpoint(dev/'checkpoints/ahem-v6-1-development-20260906/extracted/ahem-v6-1-development-20260906',seed,AHEM_NAME,EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME])
    base = base.to('cuda').eval()
    reader = AhemPhaseReader(base,phase_width=32).to('cuda').eval()
    reader.load_state_dict(torch.load(reader_path,weights_only=True,map_location='cuda')['model_state'],strict=True)
    reader.requires_grad_(False)
    gate = ObservedPriorGate().to('cpu').eval()
    gate.load_state_dict(torch.load(gate_path,weights_only=True,map_location='cpu')['model_state'],strict=True)
    datasets,owned = {},{}
    for role,paths in sources.items():
        for i,source in enumerate(paths):
            path = temporary/f'{role}-{i}.pt'
            with torch.no_grad():
                cache = prepare_cache(base,source,role,seed,source_plan,binding['source_plan_sha256'],path,
                    official=False,samples=plan['samples_per_source'][role])
                tensors = cache['tensors']
                data = project_gate(reader,tensors,plan['projection_batch'],'cuda')
                data['context'] = tensors['query_phase'].cpu()
            owned[path.name] = checked_hash(path)
            atomic_json(json.loads(path.with_suffix('.manifest.json').read_text()),output/f'{role}-{i}.manifest.json')
            # Coordinates alone contain no packet payload or future outcome.
            atomic_json({'source':source.name,'coordinates':tensors['coordinates'].tolist()},output/f'{role}-{i}-coordinates.json')
            datasets[f'{role}-{i}'] = data
            print(json.dumps({'stage':'projected','source':source.name,'samples':len(data['target'])}),flush=True)
            del cache,tensors
    train = {n:torch.cat([d[n] for k,d in datasets.items() if k.startswith('train-')]) for n in datasets['train-0']}
    snapshot = temporary/'features.pt'
    atomic_save(datasets,snapshot)
    owned[snapshot.name] = checked_hash(snapshot)
    atomic_json({'sha256':owned[snapshot.name],'rows':{k:len(d['target']) for k,d in datasets.items()},
        'features':['features','context'],'labels_only':['target'],'source_and_coordinates':'separate hash-bound projection manifests'},output/'SNAPSHOT.json')
    truth,active = labels(train,plan)
    states,decisions = {},{}
    for variant in ('summary','context'):
        x = train['features'] if variant=='summary' else torch.cat((train['features'],train['context']),1)
        states[variant] = [fit_probe(x,truth[:,f],active[:,f],plan) for f in range(2)]
        s = torch.stack([score(states[variant][f],x) for f in range(2)],1)
        thresholds = [choose_threshold(s[active[:,f],f],truth[active[:,f],f]) for f in range(2)]
        coefficients = [best_coefficient(train,f,(s[:,f]>=thresholds[f])&active[:,f],plan['coefficient_grid']) for f in range(2)]
        decisions[variant] = (thresholds,coefficients)
    decisions['constant'] = ([0.,0.],[best_coefficient(train,f,active[:,f],plan['coefficient_grid']) for f in range(2)])
    atomic_save({'states':states,'decisions':decisions,'binding':binding},output/'PROBES.pt')
    result = {}
    for name,data in (datasets | {'training-pooled':train}).items():
        if name.startswith('train-'):
            continue
        y,m = labels(data,plan)
        rows = {}
        evidence = {'active':m.tolist(),'useful':y.tolist(),'scores':{}}
        for variant in ('constant','summary','context'):
            if variant=='constant':
                s = torch.ones_like(data['target'],dtype=torch.float64)
            else:
                x = data['features'] if variant=='summary' else torch.cat((data['features'],data['context']),1)
                s = torch.stack([score(states[variant][f],x) for f in range(2)],1)
            rows[variant] = metrics(data,s,*decisions[variant],y,m)
            evidence['scores'][variant] = s.tolist()
        with torch.no_grad():
            parent = gate(*(data[n] for n in ('features','current','raw','available')))
            parent_scores = gate.network(data['features']).sigmoid()
        rows['parent_forecast'] = forecast_quality(parent.numpy(),data['target'].numpy())
        rows['parent_usefulness_auc'] = {field:auc_pairwise(y[m[:,f],f].numpy(),parent_scores[m[:,f],f].numpy()) for f,field in enumerate(('size','delay'))}
        rows['persistence_forecast'] = forecast_quality(data['current'].numpy(),data['target'].numpy())
        result[name] = rows
        atomic_json(evidence,output/f'{name}-ranking-evidence.json')
    checks = []
    for name,rows in result.items():
        if not name.startswith('validation-'):
            continue
        a,b = rows['summary']['fields']['size'],rows['context']['fields']['size']
        enough = min(a['useful'],a['not_useful']) >= plan['minimum_each_class_per_validation_source']
        checks.append({'source':name,'sufficient_classes':enough,'auc_lift':b['auc']-a['auc'] if a['auc'] is not None and b['auc'] is not None else None,
            'forecast_nonregression':rows['context']['forecast']['weighted_smooth_l1']<=rows['summary']['forecast']['weighted_smooth_l1']})
    supported = len(checks)==2 and all(c['sufficient_classes'] and c['auc_lift'] is not None and c['auc_lift']>=plan['required_auc_lift_each_source'] and c['forecast_nonregression'] for c in checks)
    atomic_json({'sources':result,'checks':checks,'context_feasibility_supported':supported,
        'interpretation':'development information probe; no promotion, causal insufficiency proof or significance claim',
        'runtime_admission':False,'assessment_opened':False},output/'RESULT.json')
    if any(checked_hash(p)!=hashes[p.name] for ps in sources.values() for p in ps) or any(checked_hash(Path(p))!=h for p,h in parents.items()):
        raise ValueError('input mutation detected')
    removed = cleanup_owned_cache(temporary,owned)
    for n in removed:
        sidecar = temporary.joinpath(n).with_suffix('.manifest.json')
        if sidecar.exists():
            sidecar.unlink()
    temporary.rmdir()
    atomic_json({'removed':removed,'temporary_directory_removed':True},output/'CLEANUP.json')
    print(json.dumps({'stage':'complete','checks':checks,'supported':supported}),flush=True)


if __name__=='__main__':
    torch.set_num_threads(4)
    run()
