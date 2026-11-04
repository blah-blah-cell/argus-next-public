"""No-training audit of inferred gate trust versus offline history usefulness."""
import json
from pathlib import Path
import numpy as np
from .ahem_population_audit import sha256
from .ahem_reader_experiment import atomic_json


def auc_pairwise(labels, scores):
    labels, scores = np.asarray(labels,dtype=bool), np.asarray(scores,dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError('aligned finite one-dimensional labels and scores required')
    positive, negative = scores[labels], scores[~labels]
    if not len(positive) or not len(negative):
        return None
    differences = positive[:,None]-negative[None,:]
    return float(((differences>0)+.5*(differences==0)).mean())


def huber(error):
    error = np.abs(error)
    return np.where(error<1,.5*error**2,error-.5)


def field_audit(current,raw,gate,target,*,minimum_span=1e-4):
    arrays = [np.asarray(a,dtype=np.float64) for a in (current,raw,gate,target)]
    current,raw,gate,target = arrays
    if any(a.shape!=current.shape or a.ndim!=1 or not np.isfinite(a).all() for a in arrays):
        raise ValueError('aligned finite fields required')
    delta = raw-current
    informative = np.abs(delta)>minimum_span
    oracle = np.clip(target,np.minimum(current,raw),np.maximum(current,raw))
    base_error,gate_error,oracle_error = huber(current-target),huber(gate-target),huber(oracle-target)
    useful = (base_error-oracle_error)>1e-8
    trust = np.zeros_like(current)
    trust[informative] = np.clip((gate[informative]-current[informative])/delta[informative],0,1)
    optimal = np.zeros_like(current)
    optimal[informative] = np.clip((target[informative]-current[informative])/delta[informative],0,1)
    mask = informative
    group = lambda m: float(trust[m].mean()) if m.any() else None
    gain, potential = base_error-gate_error,base_error-oracle_error
    positive_gain = np.maximum(gain,0)
    harm = np.maximum(-gain,0)
    return {'queries':len(current),'identifiable_gate_queries':int(mask.sum()),
        'minimum_log_span':minimum_span,'useful_among_identifiable':int((mask&useful).sum()),
        'not_useful_among_identifiable':int((mask&~useful).sum()),
        'trust_mean_when_useful':group(mask&useful),'trust_mean_when_not_useful':group(mask&~useful),
        'trust_auc_for_interval_usefulness':auc_pairwise(useful[mask],trust[mask]),
        'coefficient_mae_to_target_dependent_oracle':float(np.abs(trust[mask]-optimal[mask]).mean()) if mask.any() else None,
        'gate_helped_count':int((gain>1e-8).sum()),'gate_harmed_count':int((gain< -1e-8).sum()),
        'mean_log_huber_persistence':float(base_error.mean()),'mean_log_huber_gate':float(gate_error.mean()),
        'total_available_log_gain':float(potential.sum()),'total_realized_positive_log_gain':float(positive_gain.sum()),
        'total_introduced_log_harm':float(harm.sum()),'net_fraction_of_available_gain':float(gain.sum()/potential.sum()) if potential.sum()>0 else None,
        'per_query':{'identifiable':mask.tolist(),'useful':useful.tolist(),'trust':trust.tolist(),
            'oracle_coefficient':optimal.tolist(),'persistence_loss':base_error.tolist(),
            'gate_loss':gate_error.tolist(),'oracle_loss':oracle_error.tolist()}}


def run():
    dev = Path('../argus-dev')
    source = dev/'research-runs/ahem-prediction-capacity-20261003/round-01'
    output = dev/'research-runs/ahem-gate-usefulness-20261003/round-01'
    if output.exists():
        raise FileExistsError('immutable usefulness audit already exists')
    frozen=json.loads((source/'FROZEN.json').read_text())
    files={str(seed):source/f'{seed}.json' for seed in frozen['seeds']}
    hashes={seed:sha256(p) for seed,p in files.items()}
    output.mkdir(parents=True)
    atomic_json({'input_sha256':hashes,'source_freeze_sha256':sha256(source/'FROZEN.json'),
        'optimizer_steps':0,'role':'exploratory analysis of already used training coordinates',
        'minimum_log_span':1e-4,'minimum_huber_gain':1e-8,'assessment_access':False,
        'warning':'Targets label offline usefulness only. This is not detection AUROC, retrieval accuracy or held-out generalization.',
        'code_sha256':sha256(Path(__file__))},output/'FROZEN.json')
    results={}
    for seed,path in files.items():
        records=json.loads(path.read_text())['records']
        arrays={n:np.asarray(records[n],dtype=np.float64) for n in ('current','raw','gate','target')}
        result={name:field_audit(*(arrays[n][:,field] for n in ('current','raw','gate','target'))) for field,name in enumerate(('size','delay'))}
        atomic_json(result,output/f'{seed}.json')
        results[seed]={name:{k:v for k,v in row.items() if k!='per_query'} for name,row in result.items()}
    if any(sha256(p)!=hashes[seed] for seed,p in files.items()):
        raise ValueError('input changed during audit')
    atomic_json({'results':results,'unique_training_coordinates':256,'seeds_not_independent_coordinates':True,
        'optimizer_steps':0,'assessment_access':False},output/'RESULT.json')
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    run()
