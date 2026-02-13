from causalbo.do_calculus import SCM, E_output_given_do
from causalbo.causal_helper_funcs import subdict_with_keys, df_to_tensor
from causalbo.modules import NegateAcquisitionFunction
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement
from botorch.acquisition import PosteriorMean
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
import numpy as np
from typing import Literal
import concurrent.futures
import torch

# Multi-Information Source GP model using BoTorch
class MultiSourceGP:
    def __init__(self, sources: list[SCM], source_costs: list[int], num_obs: int):
        self.source_costs = source_costs
        self.sources = sources
        self.gps = []

        for source in sources:
            x = torch.tensor(df_to_tensor(source.observational_samples.loc[:num_obs-1, source.interventional_domain.keys()]).tolist(), dtype=torch.float64)
            y = torch.tensor([df_to_tensor(source.observational_samples.loc[:num_obs-1, source.graph.output_node]).tolist()], dtype=torch.float64).reshape(num_obs,1)

            # Create a separate GP model for each source
            model = SingleTaskGP(train_X=x,train_Y=y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)
            self.gps.append(model)


def get_acqf(model: SCM, source: SingleTaskGP, maximize: bool, f_max: torch.Tensor):

    return qKnowledgeGradient(
        model=model,
        num_fantasies=32,
        current_value=f_max,
    )

def optimize_acquisition(args):
    '''Optimize the acquisition function to find the next point to evaluate.'''
    gp, source, source_cost, maximize, f_max = args
    acqf = get_acqf(gp, source, maximize, f_max)
     
    candidate, improvement  = optimize_acqf(
            acq_function=acqf,
            bounds=torch.tensor(list(subdict_with_keys(source.interventional_domain, source.interventional_domain.keys()).values()), dtype=torch.float64).t(),
            q=1,
            num_restarts=3,
            raw_samples=30
        )

    new_x = candidate.detach()
    if not maximize:
        improvement = -1 * improvement

    return(improvement * source_cost, new_x)


def threaded_optimizer(gps: list[SingleTaskGP], sources: list[SCM], source_costs: list[int], type_trial: Literal['min','max'], f_max: torch.Tensor):

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(optimize_acquisition, (gps[s], sources[s], source_costs[s], True if type_trial == 'max' else False, f_max)) for s in range(len(gps))]

    results = [f.result() for f in futures]

    return results


def multi_source_optimization(sources: list[SCM], source_costs: list[int], ground_truth: SCM, budget: int, num_obs: int, intervention_cost: int, 
                              total_cost_standard: list[int], type_trial: Literal['min','max'], preset_opt: float = None, cutoff_criterion = "iterations", 
                              cost_cutoff = 1200 ):

    if preset_opt:
        f_max = preset_opt
    elif type_trial == 'max':
        f_max = torch.max(torch.tensor([source.observational_samples[source.graph.output_node][:num_obs].min() for source in sources])).item()
    elif type_trial == 'min':
        f_max = torch.min(torch.tensor([source.observational_samples[source.graph.output_node][:num_obs].max() for source in sources])).item()
    else:
        print('Invalid type_trial, use either "min" or "max"')
        return 1

    global_optimum_standard = f_max
    # Store optimal value assignment
    opt_value_assignment = [None] * len(sources[0].interventional_domain.keys())
    # Store changes in cost over time
    cost_over_time_standard = [0]
    # Store optimum over time
    global_optimum_over_time_standard = [f_max]
    # Standard intervention set is all non-output nodes
    cur_best = None # indicates no best source has been found yet
    multi_source_gp = MultiSourceGP(sources, source_costs, num_obs)
    iteration_count = 0
    
    cost_over_time_standard.append(sum(total_cost_standard))
    global_optimum_over_time_standard.append(global_optimum_standard)
    
    print(f'Iteration {0}: Best value so far: {f_max}')

    if cutoff_criterion == "iterations":

        for i in range(budget):

            cost_over_time_standard.append(sum(total_cost_standard))
            global_optimum_over_time_standard.append(global_optimum_standard)

            # Find the next point to evaluate by optimizing over all sources
            X_next = threaded_optimizer(multi_source_gp.gps, multi_source_gp.sources, multi_source_gp.source_costs, type_trial, f_max)

            s_idx, X_next_max = max(enumerate(X_next), key=lambda x: x[1][0])
            value_assignment = np.array(torch.flatten(X_next_max[1]))

            # Perform intervention and collect expected value
            if ground_truth.__class__.__name__ == 'ToyGraph':
                new_y = ground_truth.sample(n=1,do={var: value_assignment[idx] for idx, var in enumerate(sources[s_idx].interventional_domain.keys())})
                new_y = torch.flatten(new_y[ground_truth.graph.output_node])
                Y_next = new_y

            elif ground_truth.__class__.__name__ == 'PSAGraph':
                
                do={var: value_assignment[idx] for idx, var in enumerate(sources[s_idx].interventional_domain.keys())}
                do.update({'AGE': 55}) # Assume AGE of 55 for PSA example
                new_y = ground_truth.sample(n=200,do=do)
                new_y = torch.mean(torch.flatten(new_y[ground_truth.graph.output_node]), dim = 0, keepdim=True)
                Y_next = new_y

            elif ground_truth.__class__.__name__ == 'EcoliGraph':
                do={var: value_assignment[idx] for idx, var in enumerate(sources[s_idx].interventional_domain.keys())}
                new_y = ground_truth.sample(n=200,do=do)
                new_y = torch.mean(torch.flatten(new_y[ground_truth.graph.output_node]), dim = 0, keepdim=True)
                Y_next = new_y

            else:
                do={var: value_assignment[idx] for idx, var in enumerate(sources[s_idx].interventional_domain.keys())}
                new_y = ground_truth.sample(n=200,do=do)
                new_y = torch.mean(torch.flatten(new_y[ground_truth.graph.output_node]), dim = 0, keepdim=True)
                Y_next = new_y
            
            multi_source_gp.gps[s_idx].set_train_data(inputs=torch.cat([multi_source_gp.gps[s_idx].train_inputs[0], X_next_max[1]]), targets=torch.cat([multi_source_gp.gps[s_idx].train_targets, Y_next]), strict=False)
            fit_gpytorch_model(ExactMarginalLogLikelihood(multi_source_gp.gps[s_idx].likelihood, multi_source_gp.gps[s_idx]))

            # Update the best observed value
            if Y_next.item() < f_max and type_trial == 'min':
                f_max = Y_next.item()
                opt_value_assignment = value_assignment
                cur_best = s_idx
                global_optimum_standard = Y_next.item()

            elif Y_next.item() > f_max and type_trial == 'max':
                f_max = Y_next.item()
                opt_value_assignment = value_assignment
                cur_best = s_idx
                global_optimum_standard = Y_next.item()

            else:
                global_optimum_standard = f_max


            print(f'Iteration {i+1}: Best source: {cur_best}, Current global optimal set-value-result = {sources[cur_best].interventional_domain.keys()}: {opt_value_assignment} -> {f_max}')

            total_cost_standard[s_idx] += len(sources[s_idx].interventional_domain.keys()) * intervention_cost

        cost_over_time_standard.append(sum(total_cost_standard))
        global_optimum_over_time_standard.append(global_optimum_standard)
        
    elif cutoff_criterion == "cost":
        
        while sum(total_cost_standard) < cost_cutoff:

            cost_over_time_standard.append(sum(total_cost_standard))
            global_optimum_over_time_standard.append(global_optimum_standard)

            # Find the next point to evaluate by optimizing over all sources
            X_next = threaded_optimizer(multi_source_gp.gps, multi_source_gp.sources, multi_source_gp.source_costs, type_trial, f_max)

            s_idx, X_next_max = max(enumerate(X_next), key=lambda x: x[1][0])
            value_assignment = np.array(torch.flatten(X_next_max[1]))

            # Perform intervention and collect expected value
            if ground_truth.__class__.__name__ == 'ToyGraph':
                new_y = ground_truth.sample(n=1,do={var: value_assignment[idx] for idx, var in enumerate(sources[s_idx].interventional_domain.keys())})
                new_y = torch.flatten(new_y[ground_truth.graph.output_node])
                Y_next = new_y

            elif ground_truth.__class__.__name__ == 'PSAGraph':
                do={var: value_assignment[idx] for idx, var in enumerate(sources[s_idx].interventional_domain.keys())}
                do.update({'AGE': 55}) # Assume AGE 55 for PSA example
                new_y = ground_truth.sample(n=200,do=do)
                new_y = torch.mean(torch.flatten(new_y[ground_truth.graph.output_node]), dim = 0, keepdim=True)
                Y_next = new_y

            elif ground_truth.__class__.__name__ == 'EcoliGraph':
                do={var: value_assignment[idx] for idx, var in enumerate(sources[s_idx].interventional_domain.keys())}
                new_y = ground_truth.sample(n=200,do=do)
                new_y = torch.mean(torch.flatten(new_y[ground_truth.graph.output_node]), dim = 0, keepdim=True)
                Y_next = new_y

            else:
                # Calculate the corresponding outputs
                do={var: value_assignment[idx] for idx, var in enumerate(sources[s_idx].interventional_domain.keys())}
                new_y = ground_truth.sample(n=200,do=do)
                new_y = torch.mean(torch.flatten(new_y[ground_truth.graph.output_node]), dim = 0, keepdim=True)
                Y_next = new_y
            
            multi_source_gp.gps[s_idx].set_train_data(inputs=torch.cat([multi_source_gp.gps[s_idx].train_inputs[0], X_next_max[1]]), targets=torch.cat([multi_source_gp.gps[s_idx].train_targets, Y_next]), strict=False)
            fit_gpytorch_model(ExactMarginalLogLikelihood(multi_source_gp.gps[s_idx].likelihood, multi_source_gp.gps[s_idx]))

            # Update the best observed value
            if Y_next.item() < f_max and type_trial == 'min':
                f_max = Y_next.item()
                opt_value_assignment = value_assignment
                cur_best = s_idx
                global_optimum_standard = Y_next.item()

            elif Y_next.item() > f_max and type_trial == 'max':
                f_max = Y_next.item()
                opt_value_assignment = value_assignment
                cur_best = s_idx
                global_optimum_standard = Y_next.item()

            else:
                global_optimum_standard = f_max


            print(f'Iteration {iteration_count+1}: Best source: {cur_best}, Current global optimal set-value-result = {sources[cur_best].interventional_domain.keys()}: {opt_value_assignment} -> {f_max}')

            total_cost_standard[s_idx] += len(sources[s_idx].interventional_domain.keys()) * intervention_cost
            iteration_count += 1

    else:
        print('Invalid cutoff specification, use either "cost" or "iterations"')
        return 1
        


    return (multi_source_gp, global_optimum_over_time_standard, cost_over_time_standard)