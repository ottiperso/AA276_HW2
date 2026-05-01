# from functools import reduce 
# import numpy as np
# import os
# import sys 
# sys.path.append(os.path.join(os.path.dirname(__name__), '..')) 
# from problem4_helper import NeuralVF
# from utils.devices import find_device
# neuralvf = NeuralVF(device=find_device())

# ss_bounds = np.array(neuralvf.dynamics.state_test_range())
# ss_volume = reduce(lambda x, y: x * y, ss_bounds[:, 1] - ss_bounds[:, 0]).item() 
# print(f"State Space Volume: {ss_volume:.0f}")
# batch_size = 1000
# N = int(batch_size * 1e5)
# safe_count = 0
# import tqdm
# for _ in tqdm.tqdm(range(N//batch_size)):
#     state = neuralvf.sample_state_space(batch_size)
#     values = neuralvf.values(state)
#     safe_count += (values > 0).sum().item()
# safe_ratio = safe_count / N
# print(f"Safe Set Ratio: {safe_ratio:.3f} ({safe_count}/{N} points, {N/ss_volume * 100:.3g} % of SS sampled)")

# N = int(batch_size * 1e3)
# safe_count = 0
# import tqdm
# for _ in tqdm.tqdm(range(N//batch_size)):
#     state = neuralcbf.model.dynamics_model.sample_state_space(batch_size)
#     values = neuralcbf.values(state)
#     safe_count += (values > 0).sum().item()
# safe_ratio = safe_count / N
# print(f"Safe Set Ratio: {safe_ratio:.3f} ({safe_count}/{N} points, {N/ss_volume * 100:.3g} % of SS sampled)")

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from problem4_helper import NeuralVF

# load model
neuralvf = NeuralVF(ckpt_path='outputs/vf.ckpt')

# state space bounds from problem statement
state_lo = torch.tensor([-3,-3,-3,-1,-1,-1,-1,-5,-5,-5,-5,-5,-5], dtype=torch.float32)
state_hi = torch.tensor([ 3, 3, 3, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5], dtype=torch.float32)

# random sampling
N = 100000
batch_size = 1000
safe_count = 0

import tqdm
for _ in tqdm.tqdm(range(N // batch_size)):
    samples = torch.rand(batch_size, 13) * (state_hi - state_lo) + state_lo
    values = neuralvf.values(samples)
    safe_count += (values > 0).sum().item()

safe_ratio = safe_count / N
print(f"Neural VF safe set proportion: {safe_ratio:.4f}")