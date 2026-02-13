from pandas import DataFrame
from causalbo.do_calculus import SCM
import torch
import networkx as nx
import numpy as np
from dowhy.gcm.causal_mechanisms import StochasticModel, ConditionalStochasticModel
import torch


# Sample DAG and SCM using medical dataset provided by V. Aglietti et al.
# CausalBO does not require data to be organized in this fashion, but it does help to keep it organized in a similar manner.
class AgeModel(StochasticModel):
    def draw_samples(self, num_samples: int):
        """Draws samples for the fitted model."""
        return (55 - 75) * torch.rand(num_samples, 1) + 75
    
    def fit(self, X: np.ndarray) -> None:
        pass  # No fitting needed for known mechanism
    
    def clone(self) -> 'AgeModel':
        return AgeModel()
        
    
class AspirinModel(ConditionalStochasticModel):
    def draw_samples(self, parent_values: np.ndarray):
        age = torch.tensor(parent_values[:, 0], dtype=torch.float32)
        logits = -8.0 + 0.1 * age
        
        prob = torch.sigmoid(logits)
        return prob.unsqueeze(1).numpy()
    
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        pass  # No fitting needed for known mechanism
    
    def clone(self) -> 'AgeModel':
        return AgeModel()
    
class StatinModel(ConditionalStochasticModel):
    def draw_samples(self, parent_values: np.ndarray):
        age = torch.tensor(parent_values[:, 0], dtype=torch.float32)
        logits = -13.0 + 0.1 * age
        prob = torch.sigmoid(logits)
        return prob.unsqueeze(1).numpy()

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        pass  # No fitting needed for known mechanism
    
    def clone(self) -> 'AgeModel':
        return AgeModel()


class CancerModel(ConditionalStochasticModel):
    def draw_samples(self, parent_values: np.ndarray):
        age = torch.tensor(parent_values[:, 0], dtype=torch.float32)
        statin = torch.tensor(parent_values[:, 1], dtype=torch.float32)
        aspirin = torch.tensor(parent_values[:, 2], dtype=torch.float32)
        logits = 2.2 - 0.05 * age - 0.04 * statin + 0.02 * aspirin
        prob = torch.sigmoid(logits)
        return prob.unsqueeze(1).numpy()

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        pass  # No fitting needed for known mechanism
    
    def clone(self) -> 'AgeModel':
        return AgeModel()

class PSAModel(ConditionalStochasticModel):
    def draw_samples(self, parent_values: np.ndarray):
        statin = torch.tensor(parent_values[:, 0], dtype=torch.float32)
        aspirin = torch.tensor(parent_values[:, 1], dtype=torch.float32)
        cancer = torch.tensor(parent_values[:, 2], dtype=torch.float32)
        mean = 6.8 - 0.60 * statin + 0.55 * aspirin + 1.0 * cancer
        return torch.normal(mean, 0.2).unsqueeze(1).numpy()

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        pass  # No fitting needed for known mechanism
    
    def clone(self) -> 'AgeModel':
        return AgeModel()

class PSAGraph(object):

    def age(self, num_data_points):
        return (55 - 75) * torch.rand(num_data_points, 1) + 75

    # σ(−8.0 + 0.10 × age + 0.03 × bmi)
    def aspirin(self, input_tensor, noise_mean=0, noise_stdev=0):
            input_tensor = input_tensor[..., :1]
            new_tensor = torch.tensor([[-8.3 - 0.09 * i[0]] for i in input_tensor])
            noise = torch.normal(noise_mean, noise_stdev, new_tensor.shape)
            return torch.nn.Sigmoid()(new_tensor) + noise

    #N (6.8 + 0.04 × age − 0.15 × bmi − 0.60 × statin + 0.55 × aspirin + 1.00 × cancer, 0.4)
    def psa(self, input_tensor, noise_mean=0, noise_stdev=0):
            input_tensor = input_tensor[..., :4]
            new_tensor = torch.normal(torch.tensor([[6.9 + 0.03 * i[0] - 0.59 * i[1] + 0.46 * i[2] + 0.88 * i[3]] for i in input_tensor]), 0.4)
            noise = torch.normal(noise_mean, noise_stdev, new_tensor.shape)
            return new_tensor + noise

    # σ(−13.0 + 0.10 × age + 0.20 × bmi)
    def statin(self, input_tensor, noise_mean=0, noise_stdev=0):
            input_tensor = input_tensor[..., :1]
            new_tensor = torch.tensor([[-13.6 + 0.09 * i[0]] for i in input_tensor])
            noise = torch.normal(noise_mean, noise_stdev, new_tensor.shape)
            return torch.nn.Sigmoid()(new_tensor) + noise

    # σ(2.2 − 0.05 × age + 0.01 × bmi − 0.04 × statin + 0.02 × aspirin)
    def cancer(self, input_tensor, noise_mean=0, noise_stdev=0):
            input_tensor = input_tensor[..., :3]
            new_tensor = torch.tensor([[2.11 - 0.07 * i[0] - 0.08 * i[1] + 0.12 * i[2]] for i in input_tensor])
            noise = torch.normal(noise_mean, noise_stdev, new_tensor.shape)
            return torch.nn.Sigmoid()(new_tensor) + noise
    
    def __init__(self, num_observations = 1000, num_objective_points = None):
        # By default, use double the number of observations to train the true model.
        if num_objective_points == None:
            num_objective_points = 2 * num_observations

        self.structural_eqs = {
            "AGE": self.age,
            "ASPIRIN": self.aspirin,
            "STATIN": self.statin,
            "CANCER": self.cancer,
            "PSA": self.psa,
        }

        # Interventional domain
        self.interventional_domain = {'ASPIRIN': [0,1], 'STATIN': [0,1]}
        # Graph structure
        self.graph = SCM([
                ('AGE', 'ASPIRIN'),
                ('AGE', 'CANCER'),
                ('AGE', 'STATIN'),

                ('ASPIRIN', 'CANCER'),
                ('ASPIRIN', 'PSA'),

                ('STATIN', 'CANCER'),
                ('STATIN', 'PSA'),

                ('CANCER', 'PSA')
                ])
        
        ########################################################## Remove this for auto fitting ########################################
        self.graph.causal_model.set_causal_mechanism('AGE', AgeModel())
        self.graph.causal_model.set_causal_mechanism('ASPIRIN', AspirinModel())
        self.graph.causal_model.set_causal_mechanism('STATIN', StatinModel())
        self.graph.causal_model.set_causal_mechanism('CANCER', CancerModel())
        self.graph.causal_model.set_causal_mechanism('PSA', PSAModel())
        ################################################################################################################################


        # Generate observational data
        obs_data_age = self.age(1000)
        obs_data_aspirin = self.aspirin(obs_data_age, noise_mean=0, noise_stdev=0.2)
        obs_data_statin = self.statin(obs_data_age, noise_mean=0, noise_stdev=0.2)
        obs_data_cancer = self.cancer(torch.cat([obs_data_age, obs_data_statin, obs_data_aspirin], dim=1), noise_mean=0, noise_stdev=0.2)
        obs_data_psa = self.psa(torch.cat([obs_data_age, obs_data_statin, obs_data_aspirin, obs_data_cancer], dim=1), noise_mean=0, noise_stdev=0.2)


        # Add to dataframe
        self.observational_samples = DataFrame()
        self.observational_samples['AGE'] = torch.flatten(obs_data_age).numpy()
        self.observational_samples['ASPIRIN'] = torch.flatten(obs_data_aspirin).numpy()
        self.observational_samples['STATIN'] = torch.flatten(obs_data_statin).numpy()
        self.observational_samples['CANCER'] = torch.flatten(obs_data_cancer).numpy()
        self.observational_samples['PSA'] = torch.flatten(obs_data_psa).numpy()


        # Remove invalid samples caused by randomness
        self.observational_samples = self.observational_samples.drop(
                                        self.observational_samples[self.observational_samples['PSA'] <= 0].index)
        # Fit graph to observational data.
        self.graph.fit(self.observational_samples, init=True)

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
                    # Root node
                    values[var] = self.structural_eqs[var](n)
                else:
                    # Multiple parents supported: concatenate their values
                    input_tensor = torch.cat([values[p] for p in parents], dim=1)
                    values[var] = self.structural_eqs[var](input_tensor)

        return values


    # Wrapper for networkx draw()
    def draw(self):
        self.graph.draw()

    

        

    

