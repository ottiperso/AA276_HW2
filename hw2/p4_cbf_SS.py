import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from problem4_helper import NeuralCBF

# load model
neuralcbf = NeuralCBF(ckpt_path='outputs/cbf.ckpt')

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
    values = neuralcbf.values(samples)
    safe_count += (values > 0).sum().item()

safe_ratio = safe_count / N
print(f"Neural CBF safe set proportion: {safe_ratio:.4f}")