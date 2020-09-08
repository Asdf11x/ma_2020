import math

import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import torch

input_length = 20
mask = (torch.triu(torch.zeros(input_length, input_length)) == 1).transpose(0, 1)
mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

# print(mask)

# ax = sns.heatmap(mask, linewidth=0.5, vmin=-1.0, vmax=1.0, robust=True)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='right')
# ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right')
# plt.show()

# (padding, batch_size, features)
rand_source = torch.zeros((5, 3))
# print(rand_source)

# pad mask online
mask = (rand_source < 0.5).unsqueeze(-2)
# mask = mask.reshape(3,5,10)
print(mask.shape)
print(mask)

# pad len mask trans huso
mask_1 = (rand_source < 0.5).transpose(0, 1)
print(mask_1.shape)
print(mask_1)

mask = torch.triu(torch.ones(len(rand_source), len(rand_source)), 1)
mask = mask.masked_fill(mask == 1, float('-inf'))
print(mask.shape)
print(mask)
# tgt_len, bsz, embed_dim = rand_source.size()
tgt_len, bsz = rand_source.size()

print("\n"+ "###" * 10 + "\n")

print(f"rand_source.shape: {rand_source.shape}")
rand_source = rand_source.view(bsz, 1, tgt_len, 1)
print(f"first_view: {rand_source.shape}")
rand_source = rand_source.masked_fill(rand_source.unsqueeze(1).unsqueeze(2), float('-inf'), )
# print(rand_source)
print(rand_source.shape)
rand_source = rand_source.view(bsz * 3, tgt_len, 1)
print(rand_source.shape)
# print(torch.all(mask.eq(mask_1)))
print("\n"+ "###" * 10 + "\n")

rand_ten = torch.zeros(5,3)
print(f"rand_ten.shape: {rand_ten.shape}")

rand_ten = torch.triu(torch.ones(3, 3), 1)
rand_ten = rand_ten.masked_fill(rand_ten == 1, float('-inf'))
print(f"rand_ten_masked.shape: {rand_ten.shape}")
# print(rand_ten.unsqueeze(1).shape)
# print(rand_ten.unsqueeze(2).shape)
unsqueeze = rand_ten.masked_fill(rand_ten.unsqueeze(1).unsqueeze(2), float("-inf"),)
# print(unsqueeze)

print(unsqueeze.shape)
print("\n"+ "###" * 10 + "\n")

zeros = torch.zeros((5, 3, 4))
zeros= zeros.mean(2)
zeros[zeros == 0] = 4
print(zeros)