from pandas import DataFrame
from causalbo.do_calculus import SCM
import torch
import pandas as pd
import networkx as nx

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
        input_tensor = input_tensor[..., :2]  # [sucA, ygcE]
        new_tensor = torch.tensor([[-0.0403 + (0.2603) * i[0] + (-0.7252) * i[1]] for i in input_tensor])
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
        input_tensor = input_tensor[..., :2]  # [asnA, ygcE]
        new_tensor = torch.tensor([[-0.4155 + (0.5228) * i[0] + (-1.0585) * i[1]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.5638, new_tensor.shape)
        return new_tensor + noise

    def lacA(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :2]  # [asnA, cspG]
        new_tensor = torch.tensor([[0.4469 + (0.2724) * i[0] + (0.2539) * i[1]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 1.6991, new_tensor.shape)
        return new_tensor + noise

    def lacY(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :4]  # [asnA, cspG, eutG, lacA]
        new_tensor = torch.tensor([[-0.1149 + (-0.2040) * i[0] + (-0.2241) * i[1] + (0.3537) * i[2] + (1.0462) * i[3]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.2453, new_tensor.shape)
        return new_tensor + noise

    
    def lacZ(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :3]  # [asnA, lacA, lacY]
        new_tensor = torch.tensor([[0.2000 + (-0.0161) * i[0] + (1.3684) * i[1] + (-0.4376) * i[2]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.5838, new_tensor.shape)
        return new_tensor + noise


    def lpdA(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]  # [yedE]
        new_tensor = torch.tensor([[-0.1007 + (0.9609) * i[0]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.3740, new_tensor.shape)
        return new_tensor + noise

    
    def mopB(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :2]  # [dnaK, lacZ]
        new_tensor = torch.tensor([[0.0357 + (0.8958) * i[0] + (-0.0590) * i[1]] for i in input_tensor])
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
        input_tensor = input_tensor[..., :3]  # [cspG, pspB, yedE]
        new_tensor = torch.tensor([[-0.1901 + (0.1079) * i[0] + (0.1399) * i[1] + (-0.7456) * i[2]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev if noise_stdev > 0 else 0.4152, new_tensor.shape)
        return new_tensor + noise

    
    def pspB(self, input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :2]  # [cspG, yedE]
        new_tensor = torch.tensor([[-0.2490 + (0.2759) * i[0] + (-0.9886) * i[1]] for i in input_tensor])
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
        input_tensor = input_tensor[..., :2]  # [b1191, sucA]
        new_tensor = torch.tensor([[0.5240 + (1.8815) * i[0] + (0.6327) * i[1]] for i in input_tensor])
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

    def __init__(self, num_observations = 1000, num_objective_points = None, noise_mean = 0, noise_stdev = 0):
        # By default, use double the number of observations to train the true model.

        if num_objective_points == None:
            num_objective_points = 2 * num_observations

        self.structural_eqs = {
            "aceB": self.aceB,
            "asnA": self.asnA,
            "atpD": self.atpD,
            "atpG": self.atpG,
            "b1191": self.b1191,
            "b1583": self.b1583,
            "b1963": self.b1963,
            "cchB": self.cchB,
            "cspA": self.cspA,
            "cspG": self.cspG,
            "dnaG": self.dnaG,
            "dnaJ": self.dnaJ,
            "dnaK": self.dnaK,
            "eutG": self.eutG,
            "fixC": self.fixC,
            "flgD": self.flgD,
            "folK": self.folK,
            "ftsJ": self.ftsJ,
            "gltA": self.gltA,
            "hupB": self.hupB,
            "ibpB": self.ibpB,
            "icdA": self.icdA,
            "lacA": self.lacA,
            "lacY": self.lacY,
            "lacZ": self.lacZ,
            "lpdA": self.lpdA,
            "mopB": self.mopB,
            "nmpC": self.nmpC,
            "nuoM": self.nuoM,
            "pspA": self.pspA,
            "pspB": self.pspB,
            "sucA": self.sucA,
            "sucD": self.sucD,
            "tnaA": self.tnaA,
            "yaeM": self.yaeM,
            "yceP": self.yceP,
            "ycgX": self.ycgX,
            "yecO": self.yecO,
            "yedE": self.yedE,
            "yfaD": self.yfaD,
            "yfiA": self.yfiA,
            "ygcE": self.ygcE,
            "ygbD": self.ygbD,
            "yhdM": self.yhdM,
            "yheI": self.yheI,
            "yjbO": self.yjbO
        }
                                
        # Graph structure
        self.graph = SCM([
                            ('icdA', 'aceB'),
                            ('ygcE', 'asnA'),
                            ('sucA', 'atpD'),
                            ('ygcE', 'atpD'),
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
                            ('asnA', 'icdA'),
                            ('ygcE', 'icdA'),
                            ('asnA', 'lacA'),
                            ('cspG', 'lacA'),
                            ('asnA', 'lacY'),
                            ('cspG', 'lacY'),
                            ('eutG', 'lacY'),
                            ('lacA', 'lacY'),
                            ('asnA', 'lacZ'),
                            ('lacA', 'lacZ'),
                            ('lacY', 'lacZ'),
                            ('yedE', 'lpdA'),
                            ('dnaK', 'mopB'),
                            ('lacZ', 'mopB'),
                            ('pspA', 'nmpC'),
                            ('lacY', 'nuoM'),
                            ('cspG', 'pspA'),
                            ('pspB', 'pspA'),
                            ('yedE', 'pspA'),
                            ('cspG', 'pspB'),
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
                            ('sucA', 'ygcE'),
                            ('sucA', 'yhdM'),
                            ('atpD', 'yheI'),
                            ('yedE', 'yheI'),
                            ('fixC', 'yjbO')
                        ], 'b1583')
        
        # Same structure, deep copy
        self.true_graph = SCM([
                            ('icdA', 'aceB'),
                            ('ygcE', 'asnA'),
                            ('sucA', 'atpD'),
                            ('ygcE', 'atpD'),
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
                            ('asnA', 'icdA'),
                            ('ygcE', 'icdA'),
                            ('asnA', 'lacA'),
                            ('cspG', 'lacA'),
                            ('asnA', 'lacY'),
                            ('cspG', 'lacY'),
                            ('eutG', 'lacY'),
                            ('lacA', 'lacY'),
                            ('asnA', 'lacZ'),
                            ('lacA', 'lacZ'),
                            ('lacY', 'lacZ'),
                            ('yedE', 'lpdA'),
                            ('dnaK', 'mopB'),
                            ('lacZ', 'mopB'),
                            ('pspA', 'nmpC'),
                            ('lacY', 'nuoM'),
                            ('cspG', 'pspA'),
                            ('pspB', 'pspA'),
                            ('yedE', 'pspA'),
                            ('cspG', 'pspB'),
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
                            ('sucA', 'ygcE'),
                            ('sucA', 'yhdM'),
                            ('atpD', 'yheI'),
                            ('yedE', 'yheI'),
                            ('fixC', 'yjbO')
                        ], 'b1583')
        

        # Generate observational data
        obs_data_b1191 = self.b1191(num_objective_points )
        obs_data_eutG = self.eutG(num_objective_points )
        obs_data_cspG = self.cspG(num_objective_points )
        obs_data_sucA = self.sucA(torch.cat([obs_data_eutG], dim=1), noise_mean, noise_stdev)
        obs_data_ygcE = self.ygcE(torch.cat([obs_data_b1191,obs_data_sucA], dim=1), noise_mean, noise_stdev )
        obs_data_asnA = self.asnA(torch.cat([obs_data_ygcE], dim=1), noise_mean, noise_stdev )
        obs_data_icdA = self.icdA(torch.cat([obs_data_asnA,obs_data_ygcE], dim=1), noise_mean, noise_stdev)
        obs_data_aceB = self.aceB(torch.cat([obs_data_icdA], dim=1), noise_mean, noise_stdev )
        obs_data_atpD = self.atpD(torch.cat([obs_data_sucA,obs_data_ygcE], dim=1), noise_mean, noise_stdev )
        obs_data_atpG = self.atpG(torch.cat([obs_data_sucA], dim=1), noise_mean, noise_stdev )
        obs_data_lacA = self.lacA(torch.cat([obs_data_asnA,obs_data_cspG], dim=1), noise_mean, noise_stdev )
        obs_data_lacY = self.lacY(torch.cat([obs_data_asnA,obs_data_cspG,obs_data_eutG,obs_data_lacA], dim=1), noise_mean, noise_stdev )
        obs_data_lacZ = self.lacZ(torch.cat([obs_data_asnA,obs_data_lacA,obs_data_lacY], dim=1), noise_mean, noise_stdev)
        obs_data_fixC = self.fixC(torch.cat([obs_data_b1191], dim=1), noise_mean, noise_stdev )
        obs_data_yceP = self.yceP(torch.cat([obs_data_eutG,obs_data_fixC], dim=1), noise_mean, noise_stdev )
        obs_data_b1583 = self.b1583(torch.cat([obs_data_lacA,obs_data_lacZ,obs_data_yceP], dim=1), noise_mean, noise_stdev )
        obs_data_yedE = self.yedE(torch.cat([obs_data_cspG], dim=1), noise_mean, noise_stdev )
        obs_data_yheI = self.yheI(torch.cat([obs_data_atpD,obs_data_yedE], dim=1), noise_mean, noise_stdev )
        obs_data_b1963 = self.b1963(torch.cat([obs_data_yheI], dim=1), noise_mean, noise_stdev )
        obs_data_cchB = self.cchB(torch.cat([obs_data_fixC], dim=1), noise_mean, noise_stdev )
        obs_data_cspA = self.cspA(torch.cat([obs_data_cspG], dim=1), noise_mean, noise_stdev )
        obs_data_ycgX = self.ycgX(torch.cat([obs_data_fixC,obs_data_yheI], dim=1), noise_mean, noise_stdev )
        obs_data_dnaG = self.dnaG(torch.cat([obs_data_ycgX,obs_data_yheI], dim=1), noise_mean, noise_stdev )
        obs_data_dnaJ = self.dnaJ(torch.cat([obs_data_sucA], dim=1), noise_mean, noise_stdev )
        obs_data_dnaK = self.dnaK(torch.cat([obs_data_yheI], dim=1), noise_mean, noise_stdev )
        obs_data_flgD = self.flgD(torch.cat([obs_data_sucA], dim=1), noise_mean, noise_stdev )
        obs_data_folK = self.folK(torch.cat([obs_data_yheI], dim=1), noise_mean, noise_stdev )
        obs_data_mopB = self.mopB(torch.cat([obs_data_dnaK,obs_data_lacZ], dim=1), noise_mean, noise_stdev )
        obs_data_ftsJ = self.ftsJ(torch.cat([obs_data_mopB], dim=1), noise_mean, noise_stdev )
        obs_data_gltA = self.gltA(torch.cat([obs_data_sucA], dim=1), noise_mean, noise_stdev )
        obs_data_yfiA = self.yfiA(torch.cat([obs_data_cspA], dim=1), noise_mean, noise_stdev )
        obs_data_hupB = self.hupB(torch.cat([obs_data_cspA,obs_data_yfiA], dim=1), noise_mean, noise_stdev )
        obs_data_ibpB = self.ibpB(torch.cat([obs_data_eutG,obs_data_yceP], dim=1), noise_mean, noise_stdev )
        obs_data_lpdA = self.lpdA(torch.cat([obs_data_yedE], dim=1), noise_mean, noise_stdev )
        obs_data_pspB = self.pspB(torch.cat([obs_data_cspG,obs_data_yedE], dim=1), noise_mean, noise_stdev )
        obs_data_pspA = self.pspA(torch.cat([obs_data_cspG,obs_data_pspB,obs_data_yedE], dim=1), noise_mean, noise_stdev )
        obs_data_nmpC = self.nmpC(torch.cat([obs_data_pspA], dim=1), noise_mean, noise_stdev )
        obs_data_nuoM = self.nuoM(torch.cat([obs_data_lacY], dim=1), noise_mean, noise_stdev )
        obs_data_sucD = self.sucD(torch.cat([obs_data_sucA], dim=1), noise_mean, noise_stdev )
        obs_data_tnaA = self.tnaA(torch.cat([obs_data_b1191,obs_data_fixC,obs_data_sucA], dim=1), noise_mean, noise_stdev )
        obs_data_yaeM = self.yaeM(torch.cat([obs_data_cspG,obs_data_lacA,obs_data_lacZ], dim=1), noise_mean, noise_stdev )
        obs_data_yecO = self.yecO(torch.cat([obs_data_cspG], dim=1), noise_mean, noise_stdev )
        obs_data_yfaD = self.yfaD(torch.cat([obs_data_eutG,obs_data_sucA,obs_data_yceP], dim=1), noise_mean, noise_stdev )
        obs_data_ygbD = self.ygbD(torch.cat([obs_data_fixC], dim=1), noise_mean, noise_stdev )
        obs_data_yhdM = self.yhdM(torch.cat([obs_data_sucA], dim=1), noise_mean, noise_stdev )
        obs_data_yjbO = self.yjbO(torch.cat([obs_data_fixC], dim=1), noise_mean, noise_stdev )

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

        # Generate objective data
        obj_data_b1191 = self.b1191(num_objective_points, noise_stdev = 0.0001)
        obj_data_eutG = self.eutG(num_objective_points, noise_stdev = 0.00001)
        obj_data_cspG = self.cspG(num_objective_points, noise_stdev = 0.00001)
        obj_data_sucA = self.sucA(torch.cat([obj_data_eutG], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_ygcE = self.ygcE(torch.cat([obj_data_b1191,obj_data_sucA], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_asnA = self.asnA(torch.cat([obj_data_ygcE], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_icdA = self.icdA(torch.cat([obj_data_asnA,obj_data_ygcE], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_aceB = self.aceB(torch.cat([obj_data_icdA], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_atpD = self.atpD(torch.cat([obj_data_sucA,obj_data_ygcE], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_atpG = self.atpG(torch.cat([obj_data_sucA], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_lacA = self.lacA(torch.cat([obj_data_asnA,obj_data_cspG], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_lacY = self.lacY(torch.cat([obj_data_asnA,obj_data_cspG,obj_data_eutG,obj_data_lacA], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_lacZ = self.lacZ(torch.cat([obj_data_asnA,obj_data_lacA,obj_data_lacY], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_fixC = self.fixC(torch.cat([obj_data_b1191], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_yceP = self.yceP(torch.cat([obj_data_eutG,obj_data_fixC], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_b1583 = self.b1583(torch.cat([obj_data_lacA,obj_data_lacZ,obj_data_yceP], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_yedE = self.yedE(torch.cat([obj_data_cspG], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_yheI = self.yheI(torch.cat([obj_data_atpD,obj_data_yedE], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_b1963 = self.b1963(torch.cat([obj_data_yheI], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_cchB = self.cchB(torch.cat([obj_data_fixC], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_cspA = self.cspA(torch.cat([obj_data_cspG], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_ycgX = self.ycgX(torch.cat([obj_data_fixC,obj_data_yheI], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_dnaG = self.dnaG(torch.cat([obj_data_ycgX,obj_data_yheI], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_dnaJ = self.dnaJ(torch.cat([obj_data_sucA], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_dnaK = self.dnaK(torch.cat([obj_data_yheI], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_flgD = self.flgD(torch.cat([obj_data_sucA], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_folK = self.folK(torch.cat([obj_data_yheI], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_mopB = self.mopB(torch.cat([obj_data_dnaK,obj_data_lacZ], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_ftsJ = self.ftsJ(torch.cat([obj_data_mopB], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_gltA = self.gltA(torch.cat([obj_data_sucA], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_yfiA = self.yfiA(torch.cat([obj_data_cspA], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_hupB = self.hupB(torch.cat([obj_data_cspA,obj_data_yfiA], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_ibpB = self.ibpB(torch.cat([obj_data_eutG,obj_data_yceP], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_lpdA = self.lpdA(torch.cat([obj_data_yedE], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_pspB = self.pspB(torch.cat([obj_data_cspG,obj_data_yedE], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_pspA = self.pspA(torch.cat([obj_data_cspG,obj_data_pspB,obj_data_yedE], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_nmpC = self.nmpC(torch.cat([obj_data_pspA], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_nuoM = self.nuoM(torch.cat([obj_data_lacY], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_sucD = self.sucD(torch.cat([obj_data_sucA], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_tnaA = self.tnaA(torch.cat([obj_data_b1191,obj_data_fixC,obj_data_sucA], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_yaeM = self.yaeM(torch.cat([obj_data_cspG,obj_data_lacA,obj_data_lacZ], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_yecO = self.yecO(torch.cat([obj_data_cspG], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_yfaD = self.yfaD(torch.cat([obj_data_eutG,obj_data_sucA,obj_data_yceP], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_ygbD = self.ygbD(torch.cat([obj_data_fixC], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_yhdM = self.yhdM(torch.cat([obj_data_sucA], dim=1), noise_mean = 0, noise_stdev = 0.00001)
        obj_data_yjbO = self.yjbO(torch.cat([obj_data_fixC], dim=1), noise_mean = 0, noise_stdev = 0.00001)

        # # Add to dataframe
        self.objective_samples = DataFrame()
        self.objective_samples['aceB'] = torch.flatten(obj_data_aceB).numpy()
        self.objective_samples['asnA'] = torch.flatten(obj_data_asnA).numpy()
        self.objective_samples['atpD'] = torch.flatten(obj_data_atpD).numpy()
        self.objective_samples['atpG'] = torch.flatten(obj_data_atpG).numpy()
        self.objective_samples['b1191'] = torch.flatten(obj_data_b1191).numpy()
        self.objective_samples['b1583'] = torch.flatten(obj_data_b1583).numpy()
        self.objective_samples['b1963'] = torch.flatten(obj_data_b1963).numpy()
        self.objective_samples['cchB'] = torch.flatten(obj_data_cchB).numpy()
        self.objective_samples['cspA'] = torch.flatten(obj_data_cspA).numpy()
        self.objective_samples['cspG'] = torch.flatten(obj_data_cspG).numpy()
        self.objective_samples['dnaG'] = torch.flatten(obj_data_dnaG).numpy()
        self.objective_samples['dnaJ'] = torch.flatten(obj_data_dnaJ).numpy()
        self.objective_samples['dnaK'] = torch.flatten(obj_data_dnaK).numpy()
        self.objective_samples['eutG'] = torch.flatten(obj_data_eutG).numpy()
        self.objective_samples['fixC'] = torch.flatten(obj_data_fixC).numpy()
        self.objective_samples['flgD'] = torch.flatten(obj_data_flgD).numpy()
        self.objective_samples['folK'] = torch.flatten(obj_data_folK).numpy()
        self.objective_samples['ftsJ'] = torch.flatten(obj_data_ftsJ).numpy()
        self.objective_samples['gltA'] = torch.flatten(obj_data_gltA).numpy()
        self.objective_samples['hupB'] = torch.flatten(obj_data_hupB).numpy()
        self.objective_samples['ibpB'] = torch.flatten(obj_data_ibpB).numpy()
        self.objective_samples['icdA'] = torch.flatten(obj_data_icdA).numpy()
        self.objective_samples['lacA'] = torch.flatten(obj_data_lacA).numpy()
        self.objective_samples['lacY'] = torch.flatten(obj_data_lacY).numpy()
        self.objective_samples['lacZ'] = torch.flatten(obj_data_lacZ).numpy()
        self.objective_samples['lpdA'] = torch.flatten(obj_data_lpdA).numpy()
        self.objective_samples['mopB'] = torch.flatten(obj_data_mopB).numpy()
        self.objective_samples['nmpC'] = torch.flatten(obj_data_nmpC).numpy()
        self.objective_samples['nuoM'] = torch.flatten(obj_data_nuoM).numpy()
        self.objective_samples['pspA'] = torch.flatten(obj_data_pspA).numpy()
        self.objective_samples['pspB'] = torch.flatten(obj_data_pspB).numpy()
        self.objective_samples['sucA'] = torch.flatten(obj_data_sucA).numpy()
        self.objective_samples['sucD'] = torch.flatten(obj_data_sucD).numpy()
        self.objective_samples['tnaA'] = torch.flatten(obj_data_tnaA).numpy()
        self.objective_samples['yaeM'] = torch.flatten(obj_data_yaeM).numpy()
        self.objective_samples['yceP'] = torch.flatten(obj_data_yceP).numpy()
        self.objective_samples['ycgX'] = torch.flatten(obj_data_ycgX).numpy()
        self.objective_samples['yecO'] = torch.flatten(obj_data_yecO).numpy()
        self.objective_samples['yedE'] = torch.flatten(obj_data_yedE).numpy()
        self.objective_samples['yfaD'] = torch.flatten(obj_data_yfaD).numpy()
        self.objective_samples['yfiA'] = torch.flatten(obj_data_yfiA).numpy()
        self.objective_samples['ygbD'] = torch.flatten(obj_data_ygbD).numpy()
        self.objective_samples['ygcE'] = torch.flatten(obj_data_ygcE).numpy()
        self.objective_samples['yhdM'] = torch.flatten(obj_data_yhdM).numpy()
        self.objective_samples['yheI'] = torch.flatten(obj_data_yheI).numpy()
        self.objective_samples['yjbO'] = torch.flatten(obj_data_yjbO).numpy()

        # Fit graph to objective data.
        self.true_graph.fit(self.objective_samples, init=True)        


    def sample(self, n=1, do=None):
        """
        Generate a sample from the SCM with optional interventions.
        
        Args:
            n: number of samples
            do: dict, e.g. {"X": 1.0} or {"Z": torch.tensor([...])}
        
        Returns:
            dict of sampled variables
        """
        if do is None:
            do = {}

        values = {}

        # Establish the parent priority order as observed in the graph
        parent_order = [
        "b1191", "eutG", "cspG", "sucA", "ygcE", "asnA", "icdA", "aceB",
        "atpD", "atpG", "lacA", "lacY", "lacZ", "fixC", "yceP", "b1583",
        "yedE", "yheI", "b1963", "cchB", "cspA", "ycgX", "dnaG", "dnaJ",
        "dnaK", "flgD", "folK", "mopB", "ftsJ", "gltA", "yfiA", "hupB",
        "ibpB", "lpdA", "pspB", "pspA", "nmpC", "nuoM", "sucD", "tnaA",
        "yaeM", "yecO", "yfaD", "ygbD", "yhdM", "yjbO"
        ]

        # Topological sort of the causal graph to determine sampling order
        ordered_vars = list(nx.topological_sort(self.graph.graph))
        for var in ordered_vars:
            if var in do:
                # Intervene by setting variable to provided value
                val = do[var]
                values[var] = torch.full((n, 1), val) if not torch.is_tensor(val) else val
            else:
                # If we are not intervening in the variable, we pass the parent values
                # and calculate the child's value
                parents = list(self.graph.graph.predecessors(var))
                if not parents:
                    # Root node
                    # Structural_eqs contains the structural equations for each of the variables
                    # Exactly as provided by Aglietti et al.
                    values[var] = self.structural_eqs[var](n)
                else:
                    # Multiple parents, concatenate their values
                    parents = sorted(parents, key=lambda x: parent_order.index(x))
                    input_tensor = torch.cat([values[p] for p in parents], dim=1)
                    values[var] = self.structural_eqs[var](input_tensor)

        return values

    # Wrapper for networkx draw()
    def draw(self):
        self.graph.draw()

    

        

    

