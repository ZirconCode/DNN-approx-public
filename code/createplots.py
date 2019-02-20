
import sys
import random
import copy
import logging
import itertools
import time
import math
import os
import shutil

# from tenacity import retry
from tenacity import *
import timeout_decorator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils as utils
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.multiprocessing as mp
from torch.multiprocessing import Pool

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

import glob ##

#     np.save(dire+'resavg'+hash,avg_map) # TODO don't need hash here...
#     np.save(dire+'resnet'+hash,net_map) # needs Net class from above to load, see pickle
#     np.save(dire+'resmin'+hash,min_map)
#     np.save(dire+'resavginf'+hash,avginf_map)
#     np.save(dire+'resmininf'+hash,mininf_map)

def plot(id):
	dire = 'experiments/'+str(id)+'/'

	filepath = glob.glob('experiments/'+str(id)+'/resmininf*')[0]


	mininf_map = list(np.load(filepath))
	params = np.load(dire+'params.npy').item()
	print(params)
	print(mininf_map)

	for i, w in enumerate(params['heat_w']):
		# plot error against h
		# nans do not get plotted
		x = params['heat_h']
		y = mininf_map[i]
		ax = sns.lineplot(x=x, y=y, sort=True, lw=1, label="width"+str(w))
		

	if params['clamp_weights']:
		t = params['functionname'] +"\n"+ ', skip:' + str(params['skip_conn']) +', clamp:'+str(params['clamp'])+ ', lReLU t:' + str(params['use_leaky_to_train'])###
	else:
		t = params['functionname'] +"\n"+ ', skip:' + str(params['skip_conn']) +', no clamp '+', lReLU t:' + str(params['use_leaky_to_train'])###

	ax.set(xlabel='Layers', ylabel='infinity norm', title=t, yscale="log")
	plt.savefig('semilogy error plot inf norm'+str(id)+'.png') ###
	plt.show()

	## Plots die off due to nan and inf

# for i in range(200,210):
# 	plot(i)


# dl = np.load('experiments/600/dead_list.npy')
# ve = np.load('experiments/600/val_error_list.npy')
# # te = np.load('experiments/600/error_list.npy')
# # dll = np.load('experiments/600/dead_layer_list.npy')
# dl = dl[()]
# ve = ve[()]
# te = te[()]
# dll = dll[()]

# plt.scatter(dl,ve)
# plt.show()

# sns.boxplot(dl,ve)
# sns.swarmplot(dl,ve,color=".25")


dl = np.load('experiments/600/dead_list.npy')[()]
ve = np.load('experiments/600/val_error_list.npy')[()]
correl = np.corrcoef(dl,ve)[0,1]
ax = sns.regplot(dl, ve, x_jitter=.2, label='Simple Leaky, Corellation: '+str(correl)); #lmplot

dl = np.load('experiments/601/dead_list.npy')[()]
ve = np.load('experiments/601/val_error_list.npy')[()]
correl = np.corrcoef(dl,ve)[0,1]
ax = sns.regplot(dl, ve, x_jitter=.2, label='Simple Normal, Corellation: '+str(correl)); #lmplot

ax.legend()
ax.set(xlabel="Dead ReLU's", ylabel='Validation Error', title="Simple Experiment")
plt.savefig('simple.svg',format='svg', dpi=1200)
plt.show()
plt.close()


dl = np.load('experiments/603/dead_list.npy')[()]
ve = np.load('experiments/603/val_error_list.npy')[()]
correl = np.corrcoef(dl,ve)[0,1]
ax = sns.regplot(dl, ve, x_jitter=.2, label='Complex Normal, Corellation: '+str(correl)); #lmplot

dl = np.load('experiments/604/dead_list.npy')[()]
ve = np.load('experiments/604/val_error_list.npy')[()]
correl = np.corrcoef(dl,ve)[0,1]
ax = sns.regplot(dl, ve, x_jitter=.2, label='Complex Leaky, Corellation: '+str(correl)); #lmplot


ax.legend()
ax.set(xlabel="Dead ReLU's", ylabel='Validation Error', title="Complex Experiment")
plt.savefig('complex.svg',format='svg', dpi=1200)
plt.show()
plt.close()
