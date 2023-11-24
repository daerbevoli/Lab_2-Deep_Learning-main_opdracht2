import torch

from utilities.utils import mse

rgb = [(0.5021, 0.1138, 0.9047), (0.2843, 0.0684, 0.6829), (0.1935, 0.5483, 0.3117),
       (0.8017, 0.8733, 0.6258), (0.5914, 0.6004, 0.2893), (0.7038, 0.5983, 0.9914)]



tt = torch.FloatTensor(rgb)
print(tt)

