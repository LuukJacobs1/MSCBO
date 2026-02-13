from causalbo.modules import CausalMean, CausalRBF
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from causalbo.do_calculus import SCM, E_output_given_do
from causalbo.causal_helper_funcs import calculate_epsilon, df_to_tensor, subdict_with_keys
import numpy as np
from pandas import DataFrame, concat
from typing import Literal
from causalbo.pomis_generator import PomisGenerator
import concurrent.futures
import time

import torch
import random
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.mlls.leave_one_out_pseudo_likelihood import LeaveOneOutPseudoLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement
from botorch.optim import optimize_acqf


def CBOLoop(observational_samples: DataFrame, graph: SCM, exploration_set: list[list[str]] | None, 
            num_steps: int, num_initial_obs: int, num_obs_per_step: int, num_max_allowed_obs: int, intervention_cost: int,
            interventional_domain: dict[list[float]], type_trial: Literal['min', 'max'], pomis_criterion: Literal['distance','effect','random'],
            objective_function: SCM, early_stopping_iters: int = 0, verbose: bool = False, preset_opt: float = None, cutoff_criterion = "iterations",
            cost_cutoff = 1200):
    
    num_total_obs: int = num_initial_obs
    D_o: DataFrame = observational_samples[:num_initial_obs]
    D_i: dict[str, torch.Tensor] = {}
    GPs: dict[str, SingleTaskGP] = {}
    global_optimum: float = 0
    global_optimal_set: list[str] = ['None']
    global_optimal_value: torch.Tensor = None
    num_iters_without_improvement: int = 0
    total_cost: float = num_initial_obs
    cost_over_time: list = [0]
    iteration_count = 0

    if preset_opt:
        global_optimum = preset_opt
    elif type_trial == 'min':
        global_optimum = max(D_o[graph.output_node])
    elif type_trial == 'max':
        global_optimum = min(D_o[graph.output_node])
    else:
        print('Invalid type_trial, use either "min" or "max"')
        return

    optimum_over_time: list = [global_optimum]

    # Calculate the set of POMIS if no exploration set is passed by the User
    if exploration_set is None:
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
            print('Invalid POMIS-criterion, use either "distance", "effect", or "random"')
            return
        
        if verbose:
                print(f"POMIS: {pomis}")
                print(f"Criterion-optimal MIS {exploration_set}")
    
    for s in exploration_set:
        set_identifier = ''.join(s)
        input_dim = len(s)
        GPs[set_identifier] = SingleTaskGP(
            train_X=torch.empty(0, input_dim, dtype=torch.float64),
            train_Y=torch.empty(0, 1, dtype=torch.float64),
            covar_module=CausalRBF(
                interventional_variable=s,
                causal_model=graph),
            mean_module=CausalMean(
                interventional_variable=s,
                causal_model=graph))
        D_i[set_identifier] = torch.empty((0, len(s) + 1), dtype=torch.float64)
    

    if cutoff_criterion == "iterations":

        for t in range(num_steps):
            optimum_over_time.append(global_optimum.item() if type(global_optimum) == torch.Tensor else global_optimum)
            cost_over_time.append(total_cost)

            if(early_stopping_iters != 0 and num_iters_without_improvement > early_stopping_iters):
                print("Early stopping reached max num of iters without improvement.")
                break

            print(f"Iteration {t}")
            print(f"Current global optimal set-value-result = {global_optimal_set}: {global_optimal_value} -> {global_optimum}")

            uniform = np.random.uniform(0., 1.)
            if D_o.empty:
                epsilon = 1
            elif t == 1:
                epsilon = 0
            else:
                t0 = time.time()
                epsilon = calculate_epsilon(observational_samples=D_o, interventional_domain=interventional_domain, n_max=num_max_allowed_obs)
                t1 = time.time()
                if verbose:
                    print(f"calculation of epsilon spanned {t1-t0} seconds")

            if verbose:
                print(f'Epsilon: {epsilon} - Uniform: {uniform}')

            if(epsilon > uniform):
                print(f'Observing {num_obs_per_step} new data points.')
                num_total_obs += num_obs_per_step
                D_o = observational_samples[:num_total_obs]
                graph.fit(D_o)
                total_cost += 1 * num_obs_per_step

            else:
                print('Intervening...')
                t0 = time.time()

                solutions = {}

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(intervention_threaded, (s, GPs, type_trial, interventional_domain, global_optimum)) for s in exploration_set]

                results = [f.result() for f in futures]

                t1 = time.time()

                if verbose:
                    print(f"Intervention step spanned {t1-t0} seconds")

                for r in results:
                    solutions[r[0]] = (r[1], r[2], r[3])

                best_point = solutions[max(solutions.keys())]
                x_values = torch.flatten(best_point[1])

                total_cost += intervention_cost * len(best_point[2])

                # Perform intervention and collect expected value
                if objective_function.__class__.__name__ == 'ToyGraph':
                    new_y = objective_function.sample(n=1,do={var: x_values[idx].item() for idx, var in enumerate(best_point[2])})
                    new_y = torch.flatten(new_y[objective_function.graph.output_node])

                elif objective_function.__class__.__name__ == 'PSAGraph':

                    do = {var: x_values[idx].item() for idx, var in enumerate(best_point[2])}
                    do.update({'AGE': 55})
                    new_y = objective_function.sample(n=200,do=do)
                    new_y = torch.mean(torch.flatten(new_y[objective_function.graph.output_node]))

                elif objective_function.__class__.__name__ == 'EcoliGraph':
                    do = {var: x_values[idx].item() for idx, var in enumerate(best_point[2])}
                    new_y = objective_function.sample(n=200,do=do)
                    new_y = torch.mean(torch.flatten(new_y[objective_function.graph.output_node]))
                
                else:
                    do = {var: x_values[idx].item() for idx, var in enumerate(best_point[2])}
                    new_y = objective_function.sample(n=200,do=do)
                    new_y = torch.mean(torch.flatten(new_y[objective_function.graph.output_node]))

                if verbose:
                    print(f'Optimal set-value pair: {best_point[0]} - {x_values}')

                if verbose:
                    print(f'Updating D_i for {best_point[0]}...')

                interventional_data = D_i[best_point[0]]
                new_row = torch.cat([x_values, torch.flatten(new_y)], dim=0).unsqueeze(0)
                interventional_data = torch.cat([interventional_data, new_row], dim=0)

                if verbose:
                    print(f'Updating GP posterior for {best_point[0]}...')

                gp = GPs[best_point[0]]

                gp.set_train_data(inputs=interventional_data[:, :-1], targets=torch.flatten(interventional_data[:, -1:]), strict=False)
                        
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_model(mll)

                GPs[best_point[0]] = gp
                D_i[best_point[0]] = interventional_data

                if verbose:
                    print('Updating global optimum...')

                if (global_optimum == None or 
                (type_trial == 'max' and torch.flatten(new_y)[0] > global_optimum) or 
                (type_trial == 'min' and torch.flatten(new_y)[0] < global_optimum)):
                    global_optimum = torch.flatten(new_y)[0]
                    global_optimal_value = torch.flatten(best_point[1])
                else:
                    num_iters_without_improvement += 1

                global_optimal_set = best_point[2]

        optimum_over_time.append(global_optimum.item() if type(global_optimum) == torch.Tensor else global_optimum)
        cost_over_time.append(total_cost)

    elif cutoff_criterion == "cost":

        while total_cost < cost_cutoff / 2: # Change based on the amount of sources used

            optimum_over_time.append(global_optimum.item() if type(global_optimum) == torch.Tensor else global_optimum)
            cost_over_time.append(total_cost)

            if(early_stopping_iters != 0 and num_iters_without_improvement > early_stopping_iters):
                print("Early stopping reached max num of iters without improvement.")
                break

            print(f"Iteration {iteration_count}")
            print(f"Current global optimal set-value-result = {global_optimal_set}: {global_optimal_value} -> {global_optimum}")

            uniform = np.random.uniform(0., 1.)
            if iteration_count == 0:
                epsilon = 1
            elif iteration_count == 1:
                epsilon = 0
            else:
                t0 = time.time()
                epsilon = calculate_epsilon(observational_samples=D_o, interventional_domain=interventional_domain, n_max=num_max_allowed_obs)
                t1 = time.time()
                if verbose:
                    print(f"calculation of epsilon spanned {t1-t0} seconds")

            if verbose:
                print(f'Epsilon: {epsilon} - Uniform: {uniform}')

            if(epsilon > uniform):
                print(f'Observing {num_obs_per_step} new data points.')
                num_total_obs += num_obs_per_step
                D_o = observational_samples[:num_total_obs]
                graph.fit(D_o)
                total_cost += 1 * num_obs_per_step

            else:
                print('Intervening...')
                t0 = time.time()

                solutions = {}

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(intervention_threaded, (s, GPs, type_trial, interventional_domain, global_optimum)) for s in exploration_set]

                results = [f.result() for f in futures]

                t1 = time.time()

                if verbose:
                    print(f"Intervention step spanned {t1-t0} seconds")

                for r in results:
                    solutions[r[0]] = (r[1], r[2], r[3])

                best_point = solutions[max(solutions.keys())]
                x_values = torch.flatten(best_point[1])

                total_cost += intervention_cost * len(best_point[2])

                # Perform intervention and collect expected value
                if objective_function.__class__.__name__ == 'ToyGraph':
                    new_y = objective_function.sample(n=1,do={var: x_values[idx].item() for idx, var in enumerate(best_point[2])})
                    new_y = torch.flatten(new_y[objective_function.graph.output_node])

                elif objective_function.__class__.__name__ == 'PSAGraph':
                    do = {var: x_values[idx].item() for idx, var in enumerate(best_point[2])}
                    do.update({'AGE': 55})
                    new_y = objective_function.sample(n=200,do=do)
                    new_y = torch.mean(torch.flatten(new_y[objective_function.graph.output_node]))

                elif objective_function.__class__.__name__ == 'EcoliGraph':
                    do = {var: x_values[idx].item() for idx, var in enumerate(best_point[2])}
                    new_y = objective_function.sample(n=200,do=do)
                    new_y = torch.mean(torch.flatten(new_y[objective_function.graph.output_node]))
                
                else:
                    do = {var: x_values[idx].item() for idx, var in enumerate(best_point[2])}
                    new_y = objective_function.sample(n=200,do=do)
                    new_y = torch.mean(torch.flatten(new_y[objective_function.graph.output_node]))
                    
                if verbose:
                    print(f'Optimal set-value pair: {best_point[0]} - {x_values}')

                if verbose:
                    print(f'Updating D_i for {best_point[0]}...')

                interventional_data = D_i[best_point[0]]
                new_row = torch.cat([x_values, torch.flatten(new_y)], dim=0).unsqueeze(0)
                interventional_data = torch.cat([interventional_data, new_row], dim=0)

                if verbose:
                    print(f'Updating GP posterior for {best_point[0]}...')

                gp = GPs[best_point[0]]

                gp.set_train_data(inputs=interventional_data[:, :-1], targets=torch.flatten(interventional_data[:, -1:]), strict=False)
                        
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_model(mll)

                GPs[best_point[0]] = gp
                D_i[best_point[0]] = interventional_data

                if verbose:
                    print('Updating global optimum...')

                if (global_optimum == None or 
                (type_trial == 'max' and torch.flatten(new_y)[0] > global_optimum) or 
                (type_trial == 'min' and torch.flatten(new_y)[0] < global_optimum)):
                    global_optimum = torch.flatten(new_y)[0]
                    global_optimal_value = torch.flatten(best_point[1])
                else:
                    num_iters_without_improvement += 1

                global_optimal_set = best_point[2]
                
            iteration_count += 1

    else:
        print('Invalid cutoff specification, use either "cost" or "iterations"')
        return 1

    return (global_optimum, global_optimal_set, GPs[''.join(global_optimal_set)], D_i[''.join(global_optimal_set)], D_o, cost_over_time, optimum_over_time)

# Threaded optimization of the acquisition functions
def intervention_threaded(args):

    s, GPs, type_trial, interventional_domain, global_optimum = args
    set_identifier = ''.join(s)
    gp: SingleTaskGP = GPs[set_identifier]

    current_value = torch.tensor(global_optimum)
    acqf = qKnowledgeGradient(
            model=gp,
            num_fantasies=32,
            current_value=current_value,
        )

    candidates, improvement = optimize_acqf(
        acq_function=acqf,
        bounds=torch.tensor(list(subdict_with_keys(interventional_domain, s).values()), dtype=torch.float64).t(),
        q=1,
        num_restarts=3,
        raw_samples=30
    )

    new_x = candidates.detach()

    # Negative KG-value for minimization
    if type_trial == 'min':
        improvement = -1 * improvement

    return(improvement, set_identifier, new_x, s)


