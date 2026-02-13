from causalbo.modules import CausalMean, CausalRBF
from causalbo.do_calculus import SCM, E_output_given_do
from causalbo.causal_helper_funcs import calculate_epsilon, subdict_with_keys
from causalbo.modules import NegateAcquisitionFunction
from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement, LogNoisyExpectedImprovement
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition import PosteriorMean
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
import numpy as np
from pandas import DataFrame
from typing import Literal
from causalbo.pomis_generator import PomisGenerator
import concurrent.futures
import time
import torch
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

def CBOLoopMultiSource(observational_samples: list[DataFrame], graphs: list[SCM], exploration_sets: list[list[list[str]]] | None, 
            num_steps: int, num_initial_obs: int, observation_cost: int, intervention_cost: int, num_obs_per_step: int, num_max_allowed_obs: int, 
            interventional_domains: list[dict[list[float]]], type_trial: Literal['min', 'max'], pomis_criterion: Literal['distance','effect','random'],
            objective_function: SCM, source_costs: list[int], early_stopping_iters: int = 0, verbose: bool = False, preset_opt: float = None, 
            cutoff_criterion = "iterations", cost_cutoff = 1200):
    
    num_total_obs: list[int] = num_initial_obs
    D_o: list[DataFrame] = [observational_sample[:num_initial_obs[idx]] for idx, observational_sample in enumerate(observational_samples)]
    D_i: dict[str, torch.Tensor] = {}
    GPs: dict[str, SingleTaskGP] = {}
    global_optimum: float = 0
    global_optimal_set: list[str] = ['None']
    global_optimal_value: torch.Tensor = None
    num_iters_without_improvement: int = 0
    total_cost: float = sum(num_initial_obs)
    cost_over_time: list = [0]
    n_sources = len(exploration_sets)
    global_optimal_source = None
    iteration_count = 0

    if preset_opt:
        global_optimum = preset_opt
    elif type_trial == 'min':
        global_optimum = min([max(D_o[idx][graphs[idx].output_node].values) for idx in range(len(graphs))])
    elif type_trial == 'max':
        global_optimum = max([min(D_o[idx][graphs[idx].output_node].values) for idx in range(len(graphs))])
    else:
        raise ValueError('Invalid type_trial, use either "min" or "max"')

    optimum_over_time: list = [global_optimum]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(pomis_threaded, (graphs[idx], interventional_domains[idx], pomis_criterion, verbose)) 
                        if exploration_sets[idx] is None 
                        else exploration_sets[idx] 
                        for idx in range(n_sources)]

    exploration_sets = [f.result() if isinstance(f,concurrent.futures.Future) else f for f in futures]

    for idx, i_s in enumerate(exploration_sets):
        for s in i_s:
            set_identifier = ''.join(s) + str(idx)
            input_dim = len(s)
            train_Y_init = torch.empty(0,1,dtype=dtype,device=device)
            # if type_trial == 'min':
            #     train_Y_init = -1 * train_Y_init
            GPs[set_identifier] = SingleTaskGP(
                train_X=torch.zeros(0,input_dim,dtype=dtype,device=device),
                train_Y=train_Y_init,
                covar_module=CausalRBF(interventional_variable=s, causal_model=graphs[idx]),
                mean_module=CausalMean(interventional_variable=s, causal_model=graphs[idx])
            ).to(device)
            D_i[set_identifier] = torch.empty((0, len(s) + 1), dtype=dtype, device=device)

    if cutoff_criterion == "cost":

        while total_cost < cost_cutoff:
            optimum_over_time.append(global_optimum)
            cost_over_time.append(total_cost)

            if(early_stopping_iters != 0 and num_iters_without_improvement > early_stopping_iters):
                print('Early stopping reached max num of iters without improvement.')
                break

            print(f'Iteration {iteration_count}')
            print(f'Current global optimal set-value-result = Source {global_optimal_source}: {global_optimal_set}: {global_optimal_value} -> {global_optimum}')

            # Parallel optimization and forward pass over all sources 
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(optimization_threaded, (
                    idx, 
                    D_o[idx], 
                    interventional_domains[idx], 
                    exploration_sets[idx], 
                    graphs[idx],
                    observational_samples[idx],
                    source_costs[idx],
                    GPs,
                    type_trial,
                    num_total_obs[idx],
                    num_max_allowed_obs,
                    observation_cost,
                    iteration_count,
                    verbose,
                    num_obs_per_step,
                    global_optimum
                    )) for idx in range(n_sources)]

            results = [f.result() for f in futures]
            
            # Return a tuple for all sources containing:
            # Optimal acqf value
            # Intervention set associated with acqf value
            # Cost for the step
            # Observations done within said step
            acqf_values, interventions, costs, n_obs, intervention_flags, D_o = zip(*results)
            
            for x in range(n_sources):
                total_cost += costs[x]
                num_total_obs[x] += n_obs[x]


            if any(intervention_flags):
                global_best_source = np.argmax([acqf if intervention_flags[idx] else -np.inf for idx, acqf in enumerate(acqf_values)])
                global_best_point = interventions[global_best_source]

                x_values = torch.flatten(global_best_point[1])

                total_cost += intervention_cost * len(global_best_point[2]) 

                if objective_function.__class__.__name__ == 'ToyGraph':
                    new_y = objective_function.sample(n=1,do={var: x_values[idx].item() for idx, var in enumerate(global_best_point[2])})
                    new_y = torch.flatten(new_y[objective_function.graph.output_node])

                elif objective_function.__class__.__name__ == 'PSAGraph':
                    do = {var: x_values[idx].item() for idx, var in enumerate(global_best_point[2])}
                    do.update({'AGE': 55})
                    new_y = objective_function.sample(n=200,do=do)
                    new_y = torch.mean(torch.flatten(new_y[objective_function.graph.output_node]))
            
                elif objective_function.__class__.__name__ == 'EcoliGraph':
                    do = {var: x_values[idx].item() for idx, var in enumerate(global_best_point[2])}
                    new_y = objective_function.sample(n=200,do=do)
                    new_y = torch.mean(torch.flatten(new_y[objective_function.graph.output_node]))

                elif objective_function.__class__.__name__ == 'GeneGraph':
                    new_y = objective_function.sample(global_best_point[2],x_values.numpy())

                else:
                    do = {var: x_values[idx].item() for idx, var in enumerate(global_best_point[2])}
                    new_y = objective_function.sample(n=200,do=do)
                    new_y = torch.mean(torch.flatten(new_y[objective_function.graph.output_node]))

                if verbose:
                    print(f'Optimal source-set-value pair: source {global_best_source}, {global_best_point[0]} - {x_values}')

                if verbose:
                    print(f'Updating D_i for {global_best_point[0]}...')
                
                interventional_data = D_i[global_best_point[0]]
                new_row = torch.cat([x_values, torch.flatten(new_y)], dim=0).unsqueeze(0).to(device)
                interventional_data = torch.cat([interventional_data, new_row], dim=0).to(device)

                if verbose:
                    print(f'Updating GP posterior for {global_best_point[0]}...')


                gp = GPs[global_best_point[0]]

                gp.set_train_data(inputs=interventional_data[:, :-1], targets=torch.flatten(interventional_data[:, -1:]), strict=False)
                        
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_model(mll)

                GPs[global_best_point[0]] = gp
                D_i[global_best_point[0]] = interventional_data

                if verbose:
                    print('Updating global optimum...')

                if (global_optimum == None or 
                (type_trial == 'max' and torch.flatten(new_y)[0] > global_optimum) or 
                (type_trial == 'min' and torch.flatten(new_y)[0] < global_optimum)):
                    global_optimum = torch.flatten(new_y)[0].item()
                    global_optimal_source = str(global_best_source)
                    global_optimal_set = global_best_point[2]
                    global_optimal_value = torch.flatten(global_best_point[1])
                else:
                    num_iters_without_improvement += 1

            iteration_count += 1

    elif cutoff_criterion == "iteration":

        for t in range(num_steps):
            optimum_over_time.append(global_optimum)
            cost_over_time.append(total_cost)

            if(early_stopping_iters != 0 and num_iters_without_improvement > early_stopping_iters):
                print('Early stopping reached max num of iters without improvement.')
                break

            print(f'Iteration {t}')
            print(f'Current global optimal set-value-result = {global_optimal_set}: {global_optimal_value} -> {global_optimum}')

            # Parallel optimization and foward pass over all sources 
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(optimization_threaded, (
                    idx, 
                    D_o[idx], 
                    interventional_domains[idx], 
                    exploration_sets[idx], 
                    graphs[idx],
                    observational_samples[idx],
                    source_costs[idx],
                    GPs,
                    type_trial,
                    num_total_obs[idx],
                    num_max_allowed_obs,
                    observation_cost,
                    t,
                    verbose,
                    num_obs_per_step,
                    global_optimum
                    )) for idx in range(n_sources)]

            results = [f.result() for f in futures]
            
            # Return a tuple for all sources containing:
            # Optimal acqf value
            # Intervention set associated with acqf value
            # Cost for the step
            # Observations done within said step
            acqf_values, interventions, costs, n_obs, intervention_flags, D_o = zip(*results)

            for x in range(n_sources):
                total_cost += costs[x]
                num_total_obs[x] += n_obs[x]


            if any(intervention_flags):
                global_best_source = np.argmax([acqf if intervention_flags[idx] else -np.inf for idx, acqf in enumerate(acqf_values)])
                global_best_point = interventions[global_best_source]

                x_values = torch.flatten(global_best_point[1])

                total_cost += intervention_cost * len(global_best_point[2]) 

                if objective_function.__class__.__name__ == 'ToyGraph':
                    new_y = objective_function.sample(n=1,do={var: x_values[idx].item() for idx, var in enumerate(global_best_point[2])})
                    new_y = torch.flatten(new_y[objective_function.graph.output_node])

                elif objective_function.__class__.__name__ == 'PSAGraph':
                    do = {var: x_values[idx].item() for idx, var in enumerate(global_best_point[2])}
                    do.update({'AGE': 55})
                    new_y = objective_function.sample(n=200,do=do)
                    new_y = torch.mean(torch.flatten(new_y[objective_function.graph.output_node]))

                elif objective_function.__class__.__name__ == 'EcoliGraph':
                    do = {var: x_values[idx].item() for idx, var in enumerate(global_best_point[2])}
                    new_y = objective_function.sample(n=200,do=do)
                    new_y = torch.mean(torch.flatten(new_y[objective_function.graph.output_node]))

                else:
                    do = {var: x_values[idx].item() for idx, var in enumerate(global_best_point[2])}
                    new_y = objective_function.sample(n=200,do=do)
                    new_y = torch.mean(torch.flatten(new_y[objective_function.graph.output_node]))

                if verbose:
                    print(f'Optimal source-set-value pair: source {global_best_source}, {global_best_point[0]} - {x_values}')

                if verbose:
                    print(f'Updating D_i for {global_best_point[0]}...')
                
                interventional_data = D_i[global_best_point[0]]
                new_row = torch.cat([x_values, torch.flatten(new_y)], dim=0).unsqueeze(0)
                interventional_data = torch.cat([interventional_data, new_row], dim=0)

                if verbose:
                    print(f'Updating GP posterior for {global_best_point[0]}...')

                gp = GPs[global_best_point[0]]

                gp.set_train_data(inputs=interventional_data[:, :-1], targets=torch.flatten(interventional_data[:, -1:]), strict=False)
                        
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_model(mll)

                GPs[global_best_point[0]] = gp
                D_i[global_best_point[0]] = interventional_data

                if verbose:
                    print('Updating global optimum...')

                if (global_optimum == None or 
                (type_trial == 'max' and torch.flatten(new_y)[0] > global_optimum) or 
                (type_trial == 'min' and torch.flatten(new_y)[0] < global_optimum)):
                    global_optimum = torch.flatten(new_y)[0].item()
                    global_optimal_source = str(idx)
                    global_optimal_set = global_best_point[2]
                    global_optimal_value = torch.flatten(global_best_point[1])
                else:
                    num_iters_without_improvement += 1

        optimum_over_time.append(global_optimum)
        cost_over_time.append(total_cost)
    else:
        print('Invalid cutoff specification, use either "cost" or "iterations"')
        return 1

    if global_best_source == None:
        print("Optimization failed, maybe change the Structural Equation Models and/or interventional domains?")
        return 1
    
    return (global_optimum, global_optimal_set, global_optimal_source, GPs[''.join(global_optimal_set) + str(global_optimal_source)], D_i[''.join(global_optimal_set) + str(global_optimal_source)], D_o, cost_over_time, optimum_over_time)

def get_acqf(model: SCM, interventional_domain: dict[list[float]], s, type_trial: Literal['min', 'max'], global_optimum: float):
    

    # KG requires a scalar "current best" value
    current_value = torch.tensor(global_optimum, dtype=dtype, device=device)

    return qKnowledgeGradient(
        model=model,
        num_fantasies=32,
        current_value=current_value,
    )


def optimization_threaded(args):

    idx, D_o, interventional_domain, exploration_set, graph, observational_samples, \
    source_cost, GPs, type_trial, num_total_obs, num_max_allowed_obs, observation_cost, \
    t, verbose, num_obs_per_step, global_optimum = args

    uniform = np.random.uniform(0., 1.)

    if D_o.empty:
        e = 1
    elif t == 1:
        e = 0
    else:
        t0 = time.time()
        e = calculate_epsilon(observational_samples=D_o, interventional_domain=interventional_domain, n_max=num_max_allowed_obs)
        t1 = time.time()
        if verbose:
            print(f'Calculation of epsilon source {idx} spanned {t1-t0} seconds')
            print(f'Epsilon source {idx}: {e} - Uniform: {uniform}')

    if(e > uniform):
        print(f'Observing {num_obs_per_step} new data points for source {idx}.')
        num_total_obs += num_obs_per_step
        D_o = observational_samples[:num_total_obs]
        graph.fit(D_o)
        total_cost = observation_cost * num_obs_per_step

        return (None, None, total_cost, num_obs_per_step, False, D_o)

    else:
        print(f'Performing parallel optimization source {idx}...')
        t0 = time.time()
        solutions = {}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(acq_optim_threaded, (idx, s, GPs, type_trial, interventional_domain, source_cost, global_optimum)) for s in exploration_set]

        results = [f.result() for f in futures]

        t1 = time.time()

        if verbose:
            print(f'Intervention step spanned {t1-t0} seconds')

        for r in results:
            solutions[r[0]] = (r[1], r[2], r[3])

        acqf_value = max(solutions.keys())

        best_point = solutions[acqf_value]

        return(acqf_value, best_point, 0, num_total_obs, True, D_o)


def acq_optim_threaded(args):
    idx, s, GPs, type_trial, interventional_domain, source_cost, global_optimum = args
    set_identifier = ''.join(s) + str(idx)
    gp: SingleTaskGP = GPs[set_identifier].to(device)
    acqf = get_acqf(gp, interventional_domain, s, type_trial, global_optimum)

    bounds = torch.tensor(
        list(subdict_with_keys(interventional_domain, s).values()), 
        dtype=dtype, device=device
    ).t()

    candidates, improvement = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=1,
        num_restarts=3,
        raw_samples=30,
    )

    new_x = candidates.detach().to(device)
    if type_trial == 'min':
        improvement = -1 * improvement

    return (improvement * source_cost, set_identifier, new_x, s)

def pomis_threaded(args):
    graph, interventional_domain, pomis_criterion, verbose = args

    # Calculate the set of POMIS if no exploration set is passed by the User
    pomis_generator = PomisGenerator(graph.graph, interventional_domain, graph.output_node)
    pomis = pomis_generator.POMIS()
            
    if pomis_criterion == 'distance':
        distances = pomis_generator.calculate_distances(pomis)
        exploration_set = [pomis[min(range(len(distances)), key=lambda x : distances[x])]]
    elif pomis_criterion == 'random':
        exploration_set = [random.choice(pomis)]
    elif pomis_criterion == 'effect':
        # TODO Implement a POMIS-criterion based on set effect on query variable without performing interventions
        raise NotImplementedError
    else:
        raise Exception('Invalid POMIS-criterion, use either "distance", "effect", or "random"')
    
    if verbose:
            print(f'POMIS: {pomis}')
            print(f'Criterion-optimal MIS {exploration_set}')
    
    return exploration_set

