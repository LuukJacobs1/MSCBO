from pandas import DataFrame
from causalbo.do_calculus import SCM
import torch
import networkx as nx
from dowhy.gcm.causal_mechanisms import StochasticModel, ConditionalStochasticModel
import numpy as np


class XModel(StochasticModel):
    def draw_samples(self, num_samples: int):
        prob = torch.linspace(-5, 5, num_samples).view(-1,1)
        return prob
    
    def fit(self, X: np.ndarray) -> None:
        pass  # No fitting needed for known mechanism
    
    def clone(self) -> 'XModel':
        return XModel()
        
    
class ZModel(ConditionalStochasticModel):
    def draw_samples(self, parent_values: np.ndarray):
        X = torch.tensor(parent_values[:, 0], dtype=torch.float32)
        return (torch.exp(-X))
    
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        pass  # No fitting needed for known mechanism
    
    def clone(self) -> 'ZModel':
        return ZModel()
    
class YModel(ConditionalStochasticModel):
    def draw_samples(self, parent_values: np.ndarray):
        Z = torch.tensor(parent_values[:, 0], dtype=torch.float32)
        return ((torch.cos(Z)) - (torch.exp(-Z / 20)))
    
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        pass  # No fitting needed for known mechanism
    
    def clone(self) -> 'YModel':
        return YModel()
    

# Sample DAG and SCM using toy dataset provided by V. Aglietti et al.
# CausalBO does not require data to be organized in this fashion, but it does help to keep it organized in a similar manner.
class ToyGraph(object):
    # epsilon_X
    def X(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]
        noise = torch.normal(noise_mean, noise_stdev, input_tensor.shape)
        return input_tensor + noise

    # exp(−X) + epsilon_Z
    def Z(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]
        noise = torch.normal(noise_mean, noise_stdev, input_tensor.shape)
        return (torch.exp(-input_tensor)) + noise

    # cos(Z) − exp(−Z/20) + epsilon_Y
    def Y(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]
        noise = torch.normal(noise_mean, noise_stdev, input_tensor.shape)
        return ((torch.cos(input_tensor)) - (torch.exp(-input_tensor / 20))) + noise
    
    def __init__(self, num_observations = 4000, num_objective_points = None):
        # By default, use double the number of observations to train the true model.
        if num_objective_points == None:
            num_objective_points = 2 * num_observations

        self.structural_eqs = {
            "X": self.X,
            "Z": self.Z,
            "Y": self.Y
        }

        # Interventional domain
        self.interventional_domain = {'X': [-5,5], 'Z': [-5,20]}

        # Graph structure
        self.graph = SCM([('X', 'Z'), ('Z', 'Y')])

        # Same structure, deep copy
        self.true_graph = SCM([('X', 'Z'), ('Z', 'Y')])

        ########################################################## Remove this for auto fitting ########################################
        # self.graph.causal_model.set_causal_mechanism('X', XModel())
        # self.graph.causal_model.set_causal_mechanism('Z', ZModel())
        # self.graph.causal_model.set_causal_mechanism('Y', YModel())
        ################################################################################################################################

        # Generate observational data
        obs_data_x = self.X(torch.linspace(-5, 5, num_observations).view(-1,1), noise_stdev=1)
        obs_data_z = self.Z(obs_data_x, noise_stdev=1)
        obs_data_y = self.Y(obs_data_z, noise_stdev=1)

        # Add to dataframe
        self.observational_samples = DataFrame()
        self.observational_samples['X'] = torch.flatten(obs_data_x).numpy()
        self.observational_samples['Z'] = torch.flatten(obs_data_z).numpy()
        self.observational_samples['Y'] = torch.flatten(obs_data_y).numpy()
        
        # Shuffle dataframe into random order
        self.observational_samples.sample(frac=1)
        # Fit graph to observational data.
        self.graph.fit(self.observational_samples, init=True)

        # Generate objective data
        obs_data_x = self.X(torch.linspace(-5, 5, num_objective_points).view(-1,1))
        obs_data_z = self.Z(obs_data_x)
        obs_data_y = self.Y(obs_data_z)

        # Add to dataframe
        self.objective_samples = DataFrame()
        self.objective_samples['X'] = torch.flatten(obs_data_x).numpy()
        self.objective_samples['Z'] = torch.flatten(obs_data_z).numpy()
        self.objective_samples['Y'] = torch.flatten(obs_data_y).numpy()

        ########################################################## Remove this for auto fitting ########################################
        # self.true_graph.causal_model.set_causal_mechanism('X', XModel())
        # self.true_graph.causal_model.set_causal_mechanism('Z', ZModel())
        # self.true_graph.causal_model.set_causal_mechanism('Y', YModel())
        ################################################################################################################################

        # Fit graph to objective data.
        self.true_graph.fit(self.objective_samples, init=True)    

    def sample(self, n=1, do=None):
        """
        Generate a sample from the SCM with optional interventions.
        Args:
            n: number of samples
            do: dict, e.g. {"X": 2.0} or {"Z": torch.tensor([...])}
        Returns:
            dict of sampled variables
        """
        if do is None:
            do = {}

        values = {}
        ordered_vars = list(nx.topological_sort(self.graph.graph))

        for var in ordered_vars:
            if var in do:
                val = do[var]
                values[var] = torch.full((n, 1), val) if not torch.is_tensor(val) else val
            else:
                parents = list(self.graph.graph.predecessors(var))
                if not parents:
                    # Root node, pass dummy input
                    values[var] = self.structural_eqs[var](torch.linspace(-5, 5, n).view(-1,1))
                else:
                    input_tensor = values[parents[0]]  # assumes single parent for the toy scenario
                    values[var] = self.structural_eqs[var](input_tensor)

        return values

    # Wrapper for networkx draw()
    def draw(self):
        self.graph.draw()

    

        

    

