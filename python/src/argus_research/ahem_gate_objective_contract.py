"""Deterministic mathematical fixtures; no packet data or model training."""
import json
from pathlib import Path
import torch
from .ahem_gate_utility_targets import utility_targets,utility_loss
from .ahem_reader_experiment import atomic_json
from .ahem_population_audit import sha256
from .flow_v6_objective import dynamic_loss


def normalization_fixture():
    logits=torch.zeros(2,2,dtype=torch.float64,requires_grad=True)
    coefficient=torch.tensor([[1.,0.],[0.,0.]],dtype=torch.float64)
    regret=torch.tensor([[100.,0.],[1.,0.]],dtype=torch.float64)
    full=utility_loss(logits,coefficient,regret)
    micro=sum(utility_loss(logits[i:i+1],coefficient[i:i+1],regret[i:i+1]) for i in range(2))/2
    full_gradient=torch.autograd.grad(full,logits,retain_graph=True)[0]
    micro_gradient=torch.autograd.grad(micro,logits)[0]
    # A shared scalar logit parameter sees the sum of per-example derivatives.
    return {'whole_dataset_shared_gradient':float(full_gradient.sum()),
        'mean_singleton_batch_shared_gradient':float(micro_gradient.sum()),
        'whole_dataset_loss':float(full.detach()),'mean_batch_loss':float(micro.detach()),
        'regret_mass_ratio':100.}


def shared_parameter_fixture():
    theta=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    logits=theta.expand(1,2)
    current=torch.zeros(1,2,dtype=torch.float64)
    raw=torch.tensor([[1.,10.]],dtype=torch.float64)
    target=torch.tensor([[0.,5.5]],dtype=torch.float64)
    available=torch.ones(1,1,dtype=torch.bool)
    coefficient,regret,_=utility_targets(current,raw,target,available)
    forecast=dynamic_loss(current+logits.sigmoid()*(raw-current),target)
    teacher=utility_loss(logits,coefficient,regret)
    gf=float(torch.autograd.grad(forecast,theta,retain_graph=True)[0])
    gt=float(torch.autograd.grad(teacher,theta)[0])
    return {'forecast_gradient':gf,'utility_gradient':gt,'gradient_product':gf*gt,
        'teacher_coefficient':coefficient.tolist(),
        'forecast_change_under_negative_utility_gradient_first_order':-gf*gt,
        'interpretation':'Both fieldwise directions can be sensible while their shared-parameter sums oppose.'}


def span_fixture():
    theta=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    logits=torch.stack((theta,theta*0)).reshape(1,2)
    current=torch.zeros(1,2,dtype=torch.float64)
    raw=torch.tensor([[.001,0.]],dtype=torch.float64)
    target=torch.tensor([[.00075,0.]],dtype=torch.float64)
    coefficient,regret,_=utility_targets(current,raw,target,torch.ones(1,1,dtype=torch.bool))
    forecast=dynamic_loss(current+logits.sigmoid()*(raw-current),target)
    teacher=utility_loss(logits,coefficient,regret)
    gf=float(torch.autograd.grad(forecast,theta,retain_graph=True)[0])
    gt=float(torch.autograd.grad(teacher,theta)[0])
    return {'forecast_gradient':gf,'utility_gradient':gt,
        'gradient_magnitude_ratio':abs(gt/gf),'interpretation':'Larger decision gradient is not a forecast-quality guarantee.'}


def run():
    output=Path('../argus-dev/research-runs/ahem-objective-contract-20261004/round-01')
    if output.exists():
        raise FileExistsError('immutable mathematical audit exists')
    result={'normalization':normalization_fixture(),'shared_parameter':shared_parameter_fixture(),
        'span':span_fixture(),'optimizer_steps':0,'datasets_opened':False,
        'synthetic_training_data':False,'role':'analytic numerical counterexamples, not a data experiment'}
    assert abs(result['normalization']['whole_dataset_shared_gradient']+0.09801980198019802)<1e-12
    assert abs(result['normalization']['mean_singleton_batch_shared_gradient'])<1e-12
    assert result['shared_parameter']['gradient_product']<0
    output.mkdir(parents=True)
    atomic_json({'code_sha256':sha256(Path(__file__)),
        'utility_code_sha256':sha256(Path(__file__).with_name('ahem_gate_utility_targets.py')),
        'forecast_code_sha256':sha256(Path(__file__).with_name('flow_v6_objective.py')),
        'optimizer_steps':0,'datasets_opened':False},output/'FROZEN.json')
    atomic_json(result,output/'RESULT.json')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    run()
