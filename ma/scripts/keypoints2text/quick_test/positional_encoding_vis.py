import matplotlib.pyplot as plt
import numpy as np
import math

import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import torch

# uniform_data = np.random.rand(10, 12)
# ax = sns.heatmap(uniform_data, linewidth=0.5)
# plt.show()

sin_res = []
cos_res = []

res = []
dimensions = 64
positions = 20
zipf_number= 10000.0

## OWN IMPLEMENTATION
for i in range(1, dimensions):
    temp = []
    for d in range(1, positions):
        if i % 2 == 0:
            temp.append(d * math.sin(1 / zipf_number ** ((2 * i) / positions)))
        else:
            temp.append(d * math.cos(1 / zipf_number ** ((2 * i) / positions)))
    res.append(temp)

a = np.array([sin_res, cos_res])
print(res)
# plt.imshow(a, cmap='hot', interpolation='nearest')
# ax = sns.heatmap(res, linewidth=0.5)
# plt.show()

## IMPLEMENTATION OF TRANSFORMER


pe = torch.zeros(positions, dimensions)
position = torch.arange(0, positions, dtype=torch.float).unsqueeze(1)
div_term = torch.exp(torch.arange(0, dimensions, 2).float() * (-math.log(zipf_number) / dimensions))
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
pe = pe.unsqueeze(0).transpose(0, 1)

print(torch.tensor(res).shape)
print(pe.shape)
# pe = pe.permute(1, 2, 0)
pe = pe.reshape(positions, dimensions)
print(pe.shape)

ax = sns.heatmap(pe, linewidth=0.5, vmin=-1.0, vmax=1.0, robust=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='right')

# for index in range(len(ax.xaxis.get_ticklabels())-5):
#     print(index)
#     print(ax.xaxis.get_ticklabels()[index])
    # if not index % 10 == 0:
    # ax.xaxis.get_ticklabels()[index].set_visible(False)

# for label in ax.xaxis.get_ticklabels():
#     label.set_visible(True)

# plt.setp(ax.get_xticklabels()[::2], visible=False)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right')
ax.set(xlabel='dimension', ylabel='position')
# ax = sns.heatmap(pe, linewidth=0.5, robust=True)
# plt.show()

plt.savefig('positional_encoding_vis.pdf')


