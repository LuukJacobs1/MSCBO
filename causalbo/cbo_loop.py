from causalbo.modules import CausalMean, CausalRBF
from causalbo.do_calculus import SCM, E_output_given_do
from causalbo.causal_helper_funcs import calculate_epsilon, df_to_tensor, subdict_with_keys
import numpy as np
from pandas import DataFrame, concat
from typing import Literal
from causalbo.pomis_generator import PomisGenerator

import torch
import random
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement
from botorch.optim import optimize_acqf


def CBOLoop(observational_samples: DataFrame, graph: SCM, exploration_set: list[list[str]] | None, 
            num_steps: int, num_initial_obs: int, num_obs_per_step: int, num_max_allowed_obs: int,
            interventional_domain: dict[list[float]], type_trial: Literal['min', 'max'], pomis_criterion: Literal['distance','effect','random'],
            objective_function: SCM, early_stopping_iters: int = 0, verbose: bool = False):
    
    num_total_obs: int = num_initial_obs
    D_o: DataFrame = observational_samples[:num_initial_obs]
    D_i: dict[str, torch.Tensor] = {}
    GPs: dict[str, SingleTaskGP] = {}
    global_optimum: float = 0
    global_optimal_set: list[str] = ['None']
    global_optimal_value: torch.Tensor = None
    num_iters_without_improvement: int = 0
    total_cost: float = num_initial_obs
    optimum_over_time: list = []
    cost_over_time: list = []

    if type_trial == 'min':
        global_optimum = max(D_o[graph.output_node])
    elif type_trial == 'max':
        global_optimum = min(D_o[graph.output_node])
    else:
        print('Invalid type_trial, use either "min" or "max"')
        return
    
    # Calculate the set of POMIS if no exploration set is passed by the User
    if exploration_set is None:
        pomis_generator = PomisGenerator(graph.graph, interventional_domain)
        pomis = pomis_generator.POMIS()
                
        if pomis_criterion == 'distance':
            distances = pomis_generator.calculate_distances(pomis)
            exploration_set = [pomis[min(range(len(distances)), key=lambda x : distances[x])]]
        elif pomis_criterion == 'random':
            exploration_set = [random.choice(pomis)]
        else:
            print('Invalid POMIS-criterion, use either "distance", "effect" or "random"')
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
    
    for t in range(num_steps):
        optimum_over_time.append(global_optimum)
        cost_over_time.append(total_cost)

        if(early_stopping_iters != 0 and num_iters_without_improvement > early_stopping_iters):
            print("Early stopping reached max num of iters without improvement.")
            break

        print(f"Iteration {t}")
        print(f"Current global optimal set-value-result = {global_optimal_set}: {global_optimal_value} -> {global_optimum}")

        uniform = np.random.uniform(0., 1.)
        if t == 0:
            epsilon = 1
        elif t == 1:
            epsilon = 0
        else:
            epsilon = calculate_epsilon(observational_samples=D_o, interventional_domain=interventional_domain, n_max=num_max_allowed_obs)

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

            solutions = {}

            for s in exploration_set:
                set_identifier = ''.join(s)
                gp: SingleTaskGP = GPs[set_identifier]
                
                if type_trial == 'max':
                    acqf = ExpectedImprovement(gp, best_f=global_optimum)
                else:
                    acqf = ExpectedImprovement(gp, best_f=global_optimum, maximize=False)

                candidates, _ = optimize_acqf(
                    acq_function=acqf,
                    bounds=torch.tensor(list(subdict_with_keys(interventional_domain, s).values()), dtype=torch.float64).t(),
                    q=1,
                    num_restarts=10,
                    raw_samples=100
                )

                new_x = candidates.detach()
                improvement = acqf(candidates).item()
                solutions[improvement] = (set_identifier, new_x, s)

            best_point = solutions[max(solutions.keys())]
            x_values = torch.flatten(best_point[1])

            total_cost += 10 * len(best_point[2])

            new_y = torch.tensor([E_output_given_do(interventional_variable=best_point[2], interventional_value=np.array(x_values), causal_model=objective_function)])

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
                global_optimal_set = best_point[2]
                global_optimal_value = torch.flatten(best_point[1])
            else:
                num_iters_without_improvement += 1
    
    optimum_over_time.append(global_optimum)
    cost_over_time.append(total_cost)

    return (global_optimum, global_optimal_set, GPs[''.join(global_optimal_set)], D_i[''.join(global_optimal_set)], D_o, cost_over_time, optimum_over_time)
