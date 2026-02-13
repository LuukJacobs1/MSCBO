from pandas import DataFrame
from causalbo.do_calculus import SCM
import torch
import pandas as pd

# Sample DAG and SCM using medical dataset provided by V. Aglietti et al.
# CausalBO does not require data to be organized in this fashion, but it does help to keep it organized in a similar manner.
class EcoliGraph(object):
    def aceB(self, input_tensor, noise_mean=0, noise_stdev=0.2921):
        input_tensor = input_tensor[..., :1]  # icdA is the only input
        new_tensor = torch.tensor([[0.1324 + (1.0464) * i[0]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev, new_tensor.shape)
        return new_tensor + noise

    def asnA(self, input_tensor, noise_mean=0, noise_stdev=0.3023):
        input_tensor = input_tensor[..., :1]  # ygcE is the only input
        new_tensor = torch.tensor([[0.3494 + (0.7975) * i[0]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev, new_tensor.shape)
        return new_tensor + noise
  
    def atpD(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]  # [sucA]
        new_tensor = torch.tensor([[-0.0403 + (0.2603) * i[0]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.6427, new_tensor.shape)
        return new_tensor + noise
    
    def atpG(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]  # sucA is the only input
        new_tensor = torch.tensor([[-0.8908 + (0.6180) * i[0]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.6279, new_tensor.shape)
        return new_tensor + noise

    def b1191(self, num_data_points, noise_mean=0, noise_stdev=0):
        new_tensor = torch.tensor(1.2730 * torch.rand(num_data_points, 1))
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.7801, new_tensor.shape)
        return new_tensor + noise

    def b1583(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :3]  # [lacA, lacZ, yceP]
        new_tensor = torch.tensor([[1.3820 + (-0.2457) * i[0] + (0.3422) * i[1] + (0.2407) * i[2]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 1.0529, new_tensor.shape)
        return new_tensor + noise

    def b1963(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]  # yheI
        new_tensor = torch.tensor([[0.9649 + (1.0376) * i[0]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.6139, new_tensor.shape)
        return new_tensor + noise
  
    def cchB(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]  # fixC
        new_tensor = torch.tensor([[1.0695 + (0.6180) * i[0]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.8004, new_tensor.shape)
        return new_tensor + noise

    def cspA(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]  # cspG
        new_tensor = torch.tensor([[-0.4265 + (0.2887) * i[0]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 1.2396, new_tensor.shape)
        return new_tensor + noise

    def cspG(self, num_data_points, noise_mean=0, noise_stdev=0):
        new_tensor = torch.tensor(2.0261 * torch.rand(num_data_points, 1))
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 1.0371, new_tensor.shape)
        return new_tensor + noise

    def dnaG(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :2]  # [ycgX, yheI]
        new_tensor = torch.tensor([[0.1170 + (0.5992) * i[0] + (0.1955) * i[1]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.3608, new_tensor.shape)
        return new_tensor + noise
    
    def dnaJ(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]  # sucA
        new_tensor = torch.tensor([[0.1210 + (-0.8085) * i[0]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.7586, new_tensor.shape)
        return new_tensor + noise

    def dnaK(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]  # yheI
        new_tensor = torch.tensor([[-0.2469 + (1.0797) * i[0]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.5392, new_tensor.shape)
        return new_tensor + noise

    def eutG(self, num_data_points, noise_mean=0, noise_stdev=0):
        new_tensor = torch.tensor(1.2654 * torch.rand(num_data_points, 1))
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.8314, new_tensor.shape)
        return new_tensor + noise

    def fixC(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]  # b1191
        new_tensor = torch.tensor([[0.3165 + (0.9406) * i[0]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 1.0634, new_tensor.shape)
        return new_tensor + noise

    def flgD(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]  # [sucA]
        new_tensor = torch.tensor([[-0.5167 + (0.6362) * i[0]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.6268, new_tensor.shape)
        return new_tensor + noise

    def folK(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]  # yheI
        new_tensor = torch.tensor([[0.5641 + (0.8181) * i[0]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.3667, new_tensor.shape)
        return new_tensor + noise

    
    def ftsJ(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]  # mopB
        new_tensor = torch.tensor([[0.6738 + (0.9241) * i[0]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.3955, new_tensor.shape)
        return new_tensor + noise

    def gltA(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]  # sucA
        new_tensor = torch.tensor([[-0.9572 + (0.3790) * i[0]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.8303, new_tensor.shape)
        return new_tensor + noise

    
    def hupB(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :2]  # [cspA, yfiA]
        new_tensor = torch.tensor([[-0.1821 + (-0.2984) * i[0] + (1.3867) * i[1]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.3404, new_tensor.shape)
        return new_tensor + noise


    def ibpB(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :2]  # [eutG, yceP]
        new_tensor = torch.tensor([[-0.4227 + (1.4471) * i[0] + (0.1249) * i[1]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.6789, new_tensor.shape)
        return new_tensor + noise

    
    def icdA(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]  # [asnA, ygcE]
        new_tensor = torch.tensor([[-0.4155 + (-1.0585) * i[0]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.5638, new_tensor.shape)
        return new_tensor + noise

    def lacA(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :2]  # [asnA, cspG]
        new_tensor = torch.tensor([[0.4469 + (0.2724) * i[0] + (0.2539) * i[1]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 1.6991, new_tensor.shape)
        return new_tensor + noise

    def lacY(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :4]  # [cspG, eutG, lacA]
        new_tensor = torch.tensor([[-0.1149  + (-0.2241) * i[0] + (0.3537) * i[1] + (1.0462) * i[2]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.2453, new_tensor.shape)
        return new_tensor + noise

    
    def lacZ(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :3]  # [asnA, lacA, lacY]
        new_tensor = torch.tensor([[0.2000 + (1.3684) * i[0] + (-0.4376) * i[1]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.5838, new_tensor.shape)
        return new_tensor + noise


    def lpdA(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]  # [yedE]
        new_tensor = torch.tensor([[-0.1007 + (0.9609) * i[0]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.3740, new_tensor.shape)
        return new_tensor + noise

    
    def mopB(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]  # [dnaK]
        new_tensor = torch.tensor([[0.0357 + (-0.0590) * i[0]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.5602, new_tensor.shape)
        return new_tensor + noise


    def nmpC(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]  # [pspA]
        new_tensor = torch.tensor([[0.1688 + (-0.7690) * i[0]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.5685, new_tensor.shape)
        return new_tensor + noise

    def nuoM(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]  # [lacY]
        new_tensor = torch.tensor([[-2.0176 + (0.4083) * i[0]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.9019, new_tensor.shape)
        return new_tensor + noise

    
    def pspA(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :2]  # [pspB, yedE]
        new_tensor = torch.tensor([[-0.1901  + (0.1399) * i[0] + (-0.7456) * i[1]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.4152, new_tensor.shape)
        return new_tensor + noise

    
    def pspB(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]  # [cspG, yedE]
        new_tensor = torch.tensor([[-0.2490 + (-0.9886) * i[0]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.5202, new_tensor.shape)
        return new_tensor + noise

    
    def sucA(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]  # [eutG]
        new_tensor = torch.tensor([[0.0243 + (-1.0894) * i[0]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.8115, new_tensor.shape)
        return new_tensor + noise

    
    def sucD(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]  # [sucA]
        new_tensor = torch.tensor([[-0.5907 + (0.6830) * i[0]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.6600, new_tensor.shape)
        return new_tensor + noise

    
    def tnaA(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :3]  # [b1191, fixC, sucA]
        new_tensor = torch.tensor([[-0.3861 + (-0.5926) * i[0] + (-0.2442) * i[1] + (0.1106) * i[2]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.3102, new_tensor.shape)
        return new_tensor + noise

    
    def yaeM(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :3]  # [cspG, lacA, lacZ]
        new_tensor = torch.tensor([[0.3665 + (1.4722) * i[0] + (-0.7173) * i[1] + (0.7204) * i[2]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.8100, new_tensor.shape)
        return new_tensor + noise

    
    def yceP(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :2]  # [eutG, fixC]
        new_tensor = torch.tensor([[-0.1280 + (1.1409) * i[0] + (-0.3267) * i[1]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.4091, new_tensor.shape)
        return new_tensor + noise

    def ycgX(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :2]  # [fixC, yheI]
        new_tensor = torch.tensor([[0.1584 + (-0.2716) * i[0] + (1.2448) * i[1]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.5055, new_tensor.shape)
        return new_tensor + noise
    
    def yecO(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]  # [cspG]
        new_tensor = torch.tensor([[0.2719 + (0.7949) * i[0]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.4742, new_tensor.shape)
        return new_tensor + noise

    def yedE(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]  # [cspG]
        new_tensor = torch.tensor([[-0.1606 + (-0.6420) * i[0]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.5264, new_tensor.shape)
        return new_tensor + noise

    def yfaD(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :3]  # [eutG, sucA, yceP]
        new_tensor = torch.tensor([[0.1628 + (0.2876) * i[0] + (-0.2437) * i[1] + (0.3178) * i[2]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.4324, new_tensor.shape)
        return new_tensor + noise

    def yfiA(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]  # [cspA]
        new_tensor = torch.tensor([[-1.1928 + (0.8572) * i[0]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.5618, new_tensor.shape)
        return new_tensor + noise
    
    def ygcE(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]  # [b1191]
        new_tensor = torch.tensor([[0.5240 + (1.8815) * i[0]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.5856, new_tensor.shape)
        return new_tensor + noise

    
    def ygbD(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]  # [fixC]
        new_tensor = torch.tensor([[1.3504 + (0.6607) * i[0]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.8602, new_tensor.shape)
        return new_tensor + noise

    
    def yhdM(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]  # [sucA]
        new_tensor = torch.tensor([[0.2085 + (-0.7770) * i[0]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.7669, new_tensor.shape)
        return new_tensor + noise

    
    def yheI(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :2]  # [atpD, yedE]
        new_tensor = torch.tensor([[-0.2137 + (-0.9633) * i[0] + (0.3382) * i[1]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.3750, new_tensor.shape)
        return new_tensor + noise

    
    def yjbO(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]  # [fixC]
        new_tensor = torch.tensor([[1.5910 + (-0.0706) * i[0]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 1.3606, new_tensor.shape)
        return new_tensor + noise

    def __init__(self, num_observations = 1000, num_objective_points = None):
        # By default, use double the number of observations to train the true model.

        if num_objective_points == None:
            num_objective_points = 2 * num_observations

                                        
        # Graph structure
        self.graph = SCM([
                            ('icdA', 'aceB'),
                            ('ygcE', 'asnA'),
                            ('sucA', 'atpD'),
                            ('sucA', 'atpG'),
                            ('lacA', 'b1583'),
                            ('lacZ', 'b1583'),
                            ('yceP', 'b1583'),
                            ('yheI', 'b1963'),
                            ('fixC', 'cchB'),
                            ('cspG', 'cspA'),
                            ('ycgX', 'dnaG'),
                            ('yheI', 'dnaG'),
                            ('sucA', 'dnaJ'),
                            ('yheI', 'dnaK'),
                            ('b1191', 'fixC'),
                            ('sucA', 'flgD'),
                            ('yheI', 'folK'),
                            ('mopB', 'ftsJ'),
                            ('sucA', 'gltA'),
                            ('cspA', 'hupB'),
                            ('yfiA', 'hupB'),
                            ('eutG', 'ibpB'),
                            ('yceP', 'ibpB'),
                            ('ygcE', 'icdA'),
                            ('asnA', 'lacA'),
                            ('cspG', 'lacA'),
                            ('asnA', 'lacY'),
                            ('cspG', 'lacY'),
                            ('eutG', 'lacY'),
                            ('lacA', 'lacY'),
                            ('lacA', 'lacZ'),
                            ('lacY', 'lacZ'),
                            ('yedE', 'lpdA'),
                            ('lacZ', 'mopB'),
                            ('pspA', 'nmpC'),
                            ('lacY', 'nuoM'),
                            ('pspB', 'pspA'),
                            ('yedE', 'pspA'),
                            ('yedE', 'pspB'),
                            ('eutG', 'sucA'),
                            ('sucA', 'sucD'),
                            ('b1191', 'tnaA'),
                            ('fixC', 'tnaA'),
                            ('sucA', 'tnaA'),
                            ('cspG', 'yaeM'),
                            ('lacA', 'yaeM'),
                            ('lacZ', 'yaeM'),
                            ('eutG', 'yceP'),
                            ('fixC', 'yceP'),
                            ('fixC', 'ycgX'),
                            ('yheI', 'ycgX'),
                            ('cspG', 'yecO'),
                            ('cspG', 'yedE'),
                            ('eutG', 'yfaD'),
                            ('sucA', 'yfaD'),
                            ('yceP', 'yfaD'),
                            ('cspA', 'yfiA'),
                            ('fixC', 'ygbD'),
                            ('b1191', 'ygcE'),
                            ('sucA', 'yhdM'),
                            ('atpD', 'yheI'),
                            ('yedE', 'yheI'),
                            ('fixC', 'yjbO')
                        ], 'b1583')
     

        # Generate observational data
        obs_data_b1191 = self.b1191(num_objective_points )
        obs_data_eutG = self.eutG(num_objective_points )
        obs_data_cspG = self.cspG(num_objective_points )
        obs_data_sucA = self.sucA(torch.cat([obs_data_eutG], dim=1), noise_mean = 0 )
        obs_data_ygcE = self.ygcE(torch.cat([obs_data_b1191], dim=1), noise_mean = 0 )
        obs_data_asnA = self.asnA(torch.cat([obs_data_ygcE], dim=1), noise_mean = 0 )
        obs_data_icdA = self.icdA(torch.cat([obs_data_ygcE], dim=1), noise_mean = 0 )
        obs_data_aceB = self.aceB(torch.cat([obs_data_icdA], dim=1), noise_mean = 0 )
        obs_data_atpD = self.atpD(torch.cat([obs_data_sucA], dim=1), noise_mean = 0 )
        obs_data_atpG = self.atpG(torch.cat([obs_data_sucA], dim=1), noise_mean = 0 )
        obs_data_lacA = self.lacA(torch.cat([obs_data_asnA,obs_data_cspG], dim=1), noise_mean = 0 )
        obs_data_lacY = self.lacY(torch.cat([obs_data_asnA,obs_data_cspG,obs_data_eutG,obs_data_lacA], dim=1), noise_mean = 0 )
        obs_data_lacZ = self.lacZ(torch.cat([obs_data_lacA,obs_data_lacY], dim=1), noise_mean = 0 )
        obs_data_fixC = self.fixC(torch.cat([obs_data_b1191], dim=1), noise_mean = 0 )
        obs_data_yceP = self.yceP(torch.cat([obs_data_eutG,obs_data_fixC], dim=1), noise_mean = 0 )
        obs_data_b1583 = self.b1583(torch.cat([obs_data_lacA,obs_data_lacZ,obs_data_yceP], dim=1), noise_mean = 0 )
        obs_data_yedE = self.yedE(torch.cat([obs_data_cspG], dim=1), noise_mean = 0 )
        obs_data_yheI = self.yheI(torch.cat([obs_data_atpD,obs_data_yedE], dim=1), noise_mean = 0 )
        obs_data_b1963 = self.b1963(torch.cat([obs_data_yheI], dim=1), noise_mean = 0 )
        obs_data_cchB = self.cchB(torch.cat([obs_data_fixC], dim=1), noise_mean = 0 )
        obs_data_cspA = self.cspA(torch.cat([obs_data_cspG], dim=1), noise_mean = 0 )
        obs_data_ycgX = self.ycgX(torch.cat([obs_data_fixC,obs_data_yheI], dim=1), noise_mean = 0 )
        obs_data_dnaG = self.dnaG(torch.cat([obs_data_ycgX,obs_data_yheI], dim=1), noise_mean = 0 )
        obs_data_dnaJ = self.dnaJ(torch.cat([obs_data_sucA], dim=1), noise_mean = 0 )
        obs_data_dnaK = self.dnaK(torch.cat([obs_data_yheI], dim=1), noise_mean = 0 )
        obs_data_flgD = self.flgD(torch.cat([obs_data_sucA], dim=1), noise_mean = 0 )
        obs_data_folK = self.folK(torch.cat([obs_data_yheI], dim=1), noise_mean = 0 )
        obs_data_mopB = self.mopB(torch.cat([obs_data_lacZ], dim=1), noise_mean = 0 )
        obs_data_ftsJ = self.ftsJ(torch.cat([obs_data_mopB], dim=1), noise_mean = 0 )
        obs_data_gltA = self.gltA(torch.cat([obs_data_sucA], dim=1), noise_mean = 0 )
        obs_data_yfiA = self.yfiA(torch.cat([obs_data_cspA], dim=1), noise_mean = 0 )
        obs_data_hupB = self.hupB(torch.cat([obs_data_cspA,obs_data_yfiA], dim=1), noise_mean = 0 )
        obs_data_ibpB = self.ibpB(torch.cat([obs_data_eutG,obs_data_yceP], dim=1), noise_mean = 0 )
        obs_data_lpdA = self.lpdA(torch.cat([obs_data_yedE], dim=1), noise_mean = 0 )
        obs_data_pspB = self.pspB(torch.cat([obs_data_yedE], dim=1), noise_mean = 0 )
        obs_data_pspA = self.pspA(torch.cat([obs_data_pspB,obs_data_yedE], dim=1), noise_mean = 0 )
        obs_data_nmpC = self.nmpC(torch.cat([obs_data_pspA], dim=1), noise_mean = 0 )
        obs_data_nuoM = self.nuoM(torch.cat([obs_data_lacY], dim=1), noise_mean = 0 )
        obs_data_sucD = self.sucD(torch.cat([obs_data_sucA], dim=1), noise_mean = 0 )
        obs_data_tnaA = self.tnaA(torch.cat([obs_data_b1191,obs_data_fixC,obs_data_sucA], dim=1), noise_mean = 0 )
        obs_data_yaeM = self.yaeM(torch.cat([obs_data_cspG,obs_data_lacA,obs_data_lacZ], dim=1), noise_mean = 0 )
        obs_data_yecO = self.yecO(torch.cat([obs_data_cspG], dim=1), noise_mean = 0 )
        obs_data_yfaD = self.yfaD(torch.cat([obs_data_eutG,obs_data_sucA,obs_data_yceP], dim=1), noise_mean = 0 )
        obs_data_ygbD = self.ygbD(torch.cat([obs_data_fixC], dim=1), noise_mean = 0 )
        obs_data_yhdM = self.yhdM(torch.cat([obs_data_sucA], dim=1), noise_mean = 0 )
        obs_data_yjbO = self.yjbO(torch.cat([obs_data_fixC], dim=1), noise_mean = 0 )

        # Add to dataframe
        self.observational_samples = DataFrame()
        self.observational_samples['aceB'] = torch.flatten(obs_data_aceB).numpy()
        self.observational_samples['asnA'] = torch.flatten(obs_data_asnA).numpy()
        self.observational_samples['atpD'] = torch.flatten(obs_data_atpD).numpy()
        self.observational_samples['atpG'] = torch.flatten(obs_data_atpG).numpy()
        self.observational_samples['b1191'] = torch.flatten(obs_data_b1191).numpy()
        self.observational_samples['b1583'] = torch.flatten(obs_data_b1583).numpy()
        self.observational_samples['b1963'] = torch.flatten(obs_data_b1963).numpy()
        self.observational_samples['cchB'] = torch.flatten(obs_data_cchB).numpy()
        self.observational_samples['cspA'] = torch.flatten(obs_data_cspA).numpy()
        self.observational_samples['cspG'] = torch.flatten(obs_data_cspG).numpy()
        self.observational_samples['dnaG'] = torch.flatten(obs_data_dnaG).numpy()
        self.observational_samples['dnaJ'] = torch.flatten(obs_data_dnaJ).numpy()
        self.observational_samples['dnaK'] = torch.flatten(obs_data_dnaK).numpy()
        self.observational_samples['eutG'] = torch.flatten(obs_data_eutG).numpy()
        self.observational_samples['fixC'] = torch.flatten(obs_data_fixC).numpy()
        self.observational_samples['flgD'] = torch.flatten(obs_data_flgD).numpy()
        self.observational_samples['folK'] = torch.flatten(obs_data_folK).numpy()
        self.observational_samples['ftsJ'] = torch.flatten(obs_data_ftsJ).numpy()
        self.observational_samples['gltA'] = torch.flatten(obs_data_gltA).numpy()
        self.observational_samples['hupB'] = torch.flatten(obs_data_hupB).numpy()
        self.observational_samples['ibpB'] = torch.flatten(obs_data_ibpB).numpy()
        self.observational_samples['icdA'] = torch.flatten(obs_data_icdA).numpy()
        self.observational_samples['lacA'] = torch.flatten(obs_data_lacA).numpy()
        self.observational_samples['lacY'] = torch.flatten(obs_data_lacY).numpy()
        self.observational_samples['lacZ'] = torch.flatten(obs_data_lacZ).numpy()
        self.observational_samples['lpdA'] = torch.flatten(obs_data_lpdA).numpy()
        self.observational_samples['mopB'] = torch.flatten(obs_data_mopB).numpy()
        self.observational_samples['nmpC'] = torch.flatten(obs_data_nmpC).numpy()
        self.observational_samples['nuoM'] = torch.flatten(obs_data_nuoM).numpy()
        self.observational_samples['pspA'] = torch.flatten(obs_data_pspA).numpy()
        self.observational_samples['pspB'] = torch.flatten(obs_data_pspB).numpy()
        self.observational_samples['sucA'] = torch.flatten(obs_data_sucA).numpy()
        self.observational_samples['sucD'] = torch.flatten(obs_data_sucD).numpy()
        self.observational_samples['tnaA'] = torch.flatten(obs_data_tnaA).numpy()
        self.observational_samples['yaeM'] = torch.flatten(obs_data_yaeM).numpy()
        self.observational_samples['yceP'] = torch.flatten(obs_data_yceP).numpy()
        self.observational_samples['ycgX'] = torch.flatten(obs_data_ycgX).numpy()
        self.observational_samples['yecO'] = torch.flatten(obs_data_yecO).numpy()
        self.observational_samples['yedE'] = torch.flatten(obs_data_yedE).numpy()
        self.observational_samples['yfaD'] = torch.flatten(obs_data_yfaD).numpy()
        self.observational_samples['yfiA'] = torch.flatten(obs_data_yfiA).numpy()
        self.observational_samples['ygbD'] = torch.flatten(obs_data_ygbD).numpy()
        self.observational_samples['ygcE'] = torch.flatten(obs_data_ygcE).numpy()
        self.observational_samples['yhdM'] = torch.flatten(obs_data_yhdM).numpy()
        self.observational_samples['yheI'] = torch.flatten(obs_data_yheI).numpy()
        self.observational_samples['yjbO'] = torch.flatten(obs_data_yjbO).numpy()

        # Fit graph to observational data.
        self.graph.fit(self.observational_samples, init=True)

        # Interventional domain
        self.interventional_domain = {'lacZ': [self.observational_samples['lacZ'].min(),self.observational_samples['lacZ'].max()],
                                      'lacY': [self.observational_samples['lacY'].min(),self.observational_samples['lacY'].max()],
                                      'lacA': [self.observational_samples['lacA'].min(),self.observational_samples['lacA'].max()],
                                      'asnA': [self.observational_samples['asnA'].min(),self.observational_samples['asnA'].max()],
                                      'cspG': [self.observational_samples['cspG'].min(),self.observational_samples['cspG'].max()],
                                      'eutG': [self.observational_samples['eutG'].min(),self.observational_samples['eutG'].max()],
                                      'ygcE': [self.observational_samples['ygcE'].min(),self.observational_samples['ygcE'].max()],
                                      'sucA': [self.observational_samples['sucA'].min(),self.observational_samples['sucA'].max()],
                                      'yceP': [self.observational_samples['yceP'].min(),self.observational_samples['yceP'].max()]}
    # Wrapper for networkx draw()
    def draw(self):
        self.graph.draw()

    

        

    

