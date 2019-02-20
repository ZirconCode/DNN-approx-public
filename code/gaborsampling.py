
import sys
import random
import copy
import logging
import itertools
import time
import math
import os

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

# d = 1, g: 1->1

# satisfying tuples:
# 0.01,0.01,1 / 0.1,0.1,10
alpha = 0.01
beta = 0.01
D = 1 # 

def willtheta(t):
	pi = math.pi
	if t >= 0.25:
		return pi*0.25
	elif t <= 0.25:
		return -pi*0.25
	else:
		return (96*pi)*t**5 - (20*pi)*t**3 + ((15*pi)/8)*t

def wills(t):
	return math.sin(willtheta(t) + math.pi*0.25) 

def willc(t):
	return math.cos(willtheta(t) + math.pi*0.25) 

# bump from 0.0 to 0.5
def willg(t):
	return math.sqrt(2)*wills(t+0.25)*willc(t-0.25)

## TODO: Wilson system from above:
# def willg(t):
# 	pass

def g(t):
	if abs(t) >= 1:
		return 0
	else:
		return math.e**(-1/(1-(t**2)))

def MTg(xi,x,t): # modulete xi, translate x
	# M T g
	re = math.cos(2*math.pi*xi*t)*g(t-x)
	im = math.sin(2*math.pi*xi*t)*g(t-x)
	return re,im

# wilson basis
def wilsong(l,k,x):
	if(l == 0):
		return willg(x-(k/2))
	if(l+k % 2 == 0):
		return math.sqrt(2)*math.cos(2*math.pi*l*x)*willg(x-(k/2))
	else:
		return math.sqrt(2)*math.sin(2*math.pi*l*x)*willg(x-(k/2))


# print(MTg(2,0,0.5))

def getSample():
	x = np.random.uniform(-(D+alpha/2),(D+alpha/2))
	xi = np.random.uniform(-(1/D+beta/2),(1/D+beta/2))
	# snap to lattice
	if x % alpha > alpha/2:
		x = alpha*(1 + x//alpha)
	else:
		x = alpha*(x//alpha)

	if xi % beta > beta/2:
		xi = beta*(1 + xi//beta)
	else:
		xi = beta*(xi//beta)

	return x,xi



# x_ran = np.array(np.linspace(-20,20,num=100))

# for i in range(100):
# 	x,xi = getSample()
# 	f = lambda t : MTg(xi,x,t)

# 	res = np.array(list(map(f, x_ran)))
# 	y_ran = res[:,0] # only real part
# 	# print(x_ran)
# 	plt.plot(x_ran,y_ran) # , color='red'

# 	y_ran = res[:,1] # only real part
# 	# print(x_ran)
# 	plt.plot(x_ran,y_ran) # , color='blue'

# plt.title('Uniform sampling of modulations and translations of g on a lattice')
# plt.savefig('bumpfuns8.svg',format='svg', dpi=1200) # !slow with so many markers
# plt.show()
# plt.close()

x_ran = np.array(np.linspace(-0.2,0.7,num=10000))
for l in range(-5,6,1):
	k = 0
	f = lambda t : wilsong(l,k,t)
	y_ran = np.array(list(map(f, x_ran)))
	plt.plot(x_ran,y_ran)
plt.title('g for k=0, l=-5,-4,...,5')
plt.savefig('wilsongraph.svg',format='svg', dpi=1200) # !slow with so many markers
plt.show()
plt.close()


x_ran = np.array(np.linspace(-0.1,0.6,num=100000))
y_ran = np.array(list(map(willg, x_ran)))

# print(x_ran)
# print(y_ran)
# plt.plot(x_ran,y_ran) # , color='red'
# plt.title('g')
# plt.savefig('willg.svg',format='svg', dpi=1200)
# plt.show()

