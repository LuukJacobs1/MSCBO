from pandas import DataFrame
from causalbo.do_calculus import SCM
import torch
import networkx as nx

# Sample SCM provided by Bareinboim et al. Aribitrary normal node connections
class BiGraph(object):
    # epsilon_X

    def S(self, num_data_points, noise_mean=0, noise_stdev=0):
        # Root node — exogenous
        return (0 - 10) * torch.rand(num_data_points, 1) + 10

    def W(self, input_tensor, noise_mean=0, noise_stdev=0):
        # Linear mix: W = 0.6*S + 0.4*B + ε
        input_tensor = input_tensor[..., :2]
        new_tensor = torch.tensor([[0.6 * i[0] + 0.4*i[1]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev, new_tensor.shape)
        return new_tensor + noise

    def B(self, num_data_points, noise_mean=0, noise_stdev=0):
        # Root node — exogenous
        noise = torch.normal(noise_mean, noise_stdev, (num_data_points,1))
        return noise

    def Z(self, input_tensor, noise_mean=0, noise_stdev=0):
        # Linear: Z = 1.5*Q + ε
        new_tensor = 1.5 * input_tensor
        noise = torch.normal(noise_mean, noise_stdev, new_tensor.shape)
        return new_tensor + noise

    def Q(self, num_data_points, noise_mean=0, noise_stdev=0):
        # Root node — exogenous
        noise = torch.normal(noise_mean, noise_stdev, (num_data_points,1))
        return noise

    def X(self, input_tensor, noise_mean=0, noise_stdev=0):
        # Slight nonlinearity: X = tanh(0.5*Z + 0.3*B) + ε
        input_tensor = input_tensor[..., :2]
        new_tensor = torch.tensor([[0.5 * i[0] + 0.3 * i[1]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev, new_tensor.shape)
        return torch.nn.Tanh()(new_tensor) + noise

    def T(self, num_data_points, noise_mean=0, noise_stdev=0):
        # Root node — exogenous
       return (0 - 30) * torch.rand(num_data_points, 1) + 30

    def Y(self, input_tensor, noise_mean=0, noise_stdev=0):
        # Sigmoid combination: Y = sigmoid(0.8*W + 0.5*T + 1.0*X + 0.6*B) + ε
        input_tensor = input_tensor[..., :4]
        new_tensor = torch.tensor([[0.8 * i[0] + 0.5 * i[1] + 1.0 * i[2] + 0.6 * i[3]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev, new_tensor.shape)
        return torch.nn.Sigmoid()(new_tensor) + noise
    
    def __init__(self, num_observations = 100, num_objective_points = None, noise_mean = 0, noise_stdev = 0):
        # By default, use double the number of observations to train the true model.
        if num_objective_points == None:
            num_objective_points = 2 * num_observations

        self.structural_eqs = {
        "X": self.X,
        "Z": self.Z,
        "Y": self.Y,
        "Q": self.Q,
        "W": self.W,
        "B": self.B,
        "T": self.T,
        "S": self.S,
        }

        # Graph structure
        self.graph = SCM([('S','W'), ('W', 'Y'), ('B', 'W'), ('B','X'), ('T','Y'), ('Z','X'), ('X','Y'), ('Q','Z'), ('Q','Y')])

        # Same structure, deep copy
        self.true_graph = SCM([('S','W'), ('W', 'Y'), ('B', 'W'), ('B','X'), ('T','Y'), ('Z','X'), ('X','Y'), ('Q','Z'), ('Q','Y')])

        self.dist_to_query = {'T': 1, 'W': 1, 'X': 1, 'S': 2, 'Z': 2}

        # Generate observational data
        obs_data_q = self.Q(num_observations, noise_mean = noise_mean, noise_stdev = noise_stdev)
        obs_data_s = self.S(num_observations, noise_mean = noise_mean, noise_stdev = noise_stdev)
        obs_data_t = self.T(num_observations, noise_mean = noise_mean, noise_stdev = noise_stdev)
        obs_data_b = self.B(num_observations, noise_mean = noise_mean, noise_stdev = noise_stdev)
        obs_data_w = self.W(torch.cat([obs_data_s, obs_data_b], dim=1), noise_mean = noise_mean, noise_stdev=noise_stdev)
        obs_data_z = self.Z(obs_data_q, noise_mean = noise_mean, noise_stdev=noise_stdev)
        obs_data_x = self.X(torch.cat([obs_data_z, obs_data_b], dim=1), noise_mean = noise_mean, noise_stdev=noise_stdev)
        obs_data_y = self.Y(torch.cat([obs_data_w, obs_data_t, obs_data_x, obs_data_q], dim=1), noise_mean = noise_mean, noise_stdev=noise_stdev)



        # Add to dataframe
        self.observational_samples = DataFrame()
        self.observational_samples['X'] = torch.flatten(obs_data_x).numpy()
        self.observational_samples['Z'] = torch.flatten(obs_data_z).numpy()
        self.observational_samples['Y'] = torch.flatten(obs_data_y).numpy()
        self.observational_samples['Q'] = torch.flatten(obs_data_q).numpy()
        self.observational_samples['W'] = torch.flatten(obs_data_w).numpy()
        self.observational_samples['B'] = torch.flatten(obs_data_b).numpy()
        self.observational_samples['S'] = torch.flatten(obs_data_s).numpy()
        self.observational_samples['T'] = torch.flatten(obs_data_t).numpy()

        # Interventional domain
        self.interventional_domain = {'T': [0, 30], 
                                    'W': [self.observational_samples['W'].min(), self.observational_samples['W'].max()],
                                    'X': [self.observational_samples['X'].min(), self.observational_samples['X'].max()],
                                    'S': [0, 10],
                                    'Z': [self.observational_samples['Z'].min(), self.observational_samples['Z'].max()]}

        # Shuffle dataframe into random order
        self.observational_samples.sample(frac=1)
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
                    values[var] = self.structural_eqs[var](n)
                else:
                    # Multiple parents supported: concatenate their values
                    input_tensor = torch.cat([values[p] for p in parents], dim=1)
                    values[var] = self.structural_eqs[var](input_tensor)

        return values

    # Wrapper for networkx draw()
    def draw(self):
        self.graph.draw()

    

        

    

