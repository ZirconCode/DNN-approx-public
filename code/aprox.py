

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

# Logging
# note: python logging behaves not like print, so must pass tuples...
# always appends to old log*!
log = logging.getLogger()
log.setLevel(logging.DEBUG)

log_stdout = logging.StreamHandler()
log_stdout.setLevel(logging.INFO)
log.addHandler(log_stdout)

dt = datetime.now()
ms = dt.microsecond
txt = str(dt).split('.')[0].replace(':','-')

pname = mp.current_process().name

log_file = logging.FileHandler('log/log-'+pname+'-'+txt+'.txt')
log_file.setLevel(logging.INFO)
log.addHandler(log_file)


log.info("Versions:")
log.info(("torch:",torch.__version__)) ## 0.4.1
log.info(("python:",sys.version))

 # torch.backends.cudnn.deterministic = True
log.info("-----")

params = {}

params['experiment_id'] = 900 # give a number for saving/loading the data
params['reset_experiment'] = False # !Deletes old experiment of that id!, shouldn't reset when multipooling...

params['seed'] = 4

params['lr'] = 0.1 # learning rate, 0.03 sometimes


params['clamp_weights'] = True # clamp_weights to +/- n
params['clamp'] = 4

# params['heatmap_w'] = 4 # no longer relevant, see exploreHeatmapt()
# params['heatmap_d'] = 4

    # h = [2,4,6,8,14,18,24] # tiny exp.
    # w = [2,4,8,15,16,17,20,30] # example
params['heat_h'] = [4,8,12] #[2,4,6,8,14,18,24] #[2,4]
params['heat_w'] = [6] #[2,4,8,15,16,17,20,30] #[5,6]

params['spp'] = 10 # samples per pixel for heatmap, average and min taken

params['batch_size'] = 200
params['dat_size'] = 100000
params['train_ratio'] = 0.65
params['valid_ratio'] = 0.05
# params['validation_size'] calculated below

params['decay_var'] = 0.5 

params['optSampleTimeout'] = 60*60*6 # secs. in case pool worker hangs itself...

params['max_epochs'] = 7000 # -1 for ignore
params['min_epochs'] = 3000 # to avoid early stopping

params['zero_error'] = False # partially implemented

params['mach_eps'] = 10**-13 # patience threshold ..

params['dim_in'] = 0 # set/reset by function choice below
params['dim_out'] = 0

params['break_early_patience'] = 14
params['lr_plateau_patience'] = 3
params['validation_intervals'] = 10

params['use_leaky_to_train'] = True # Use Leaky ReLU's to train, switch back after

params['test_samples'] = 100000 # to create size of uniform testing dataset

# use 1d convolutions (kern = 1) instead of affine
# '1dconv' , 'affine'
params['structure'] = 'affine' # 1dconv, not implemented
params['skip_conn'] = True # skip connections -> 2 ReLU blocks, +x

# use gpu acceleration
params['cuda'] = False # warning: not uniform
params['cuda_count'] = 1 # 
# torch.cuda.is_available()
# cuda0 = torch.device('cuda:0')
if(params['cuda']):
    params['device'] = torch.device('cuda:0') # default gpu, todo: 'cuda:0'
else:
    params['device'] = torch.device('cpu') # !

# 1 = defualt
params['workers'] = 24 # cpu multiprocessing 
# https://pytorch.org/docs/stable/notes/cuda.html
#  torch.set_num_threads(int)
# https://pytorch.org/docs/stable/notes/multiprocessing.html

params['dl_workers'] = 0 # deadlocks -> gone. see iter. , 
# care for third party libraries here...
# 0 if pooling else Error: daemonic processes are not allowed to have children
## fixed by removing graphing in children and changing spawn process to fork or server

params['functionname'] = "none"
# set/reset by relevant generator function

log.info(("params:",params))


log.info("-----")


log.info(("Seed for numpy, torch, random:", params['seed']))

def setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # for cuda, fails silently otherwise



log.info("-----")



# ==== Functions ====

def data_quadratic():
    dat_size = params['dat_size']
    x_dat = torch.rand(dat_size) # (0,1)
    y_dat = x_dat*x_dat

    test_x_dat = torch.linspace(0, 1, steps=params['test_samples']) # [0,1]
    test_y_dat = test_x_dat*test_x_dat

    # shuffle
    p = np.random.permutation(dat_size)
    x_dat = x_dat[p]
    y_dat = y_dat[p]

    params['functionname'] = 'f(x)=x^2 on (0,1)'
    params['dim_in'] = 1
    params['dim_out'] = 1

    return x_dat, y_dat, test_x_dat, test_y_dat

def data_product():
    D = 3
    dat_size = params['dat_size']
    x_dat = (torch.rand((dat_size,2))*2*D)-D
    y_dat = x_dat[:,0]*x_dat[:,1]
    # y_dat.view(-1,1)

    a = torch.linspace(-D,D,steps = int(math.sqrt(params['test_samples'])))
    b,c = torch.meshgrid([a,a])
    d = np.array(list(zip(b.flatten(),c.flatten()))) # hmm
    test_x_dat = d # [-D,D]^2
    test_y_dat = test_x_dat[:,0]*test_x_dat[:,1]

    # log.info(x_dat)
    # log.info(y_dat)

    params['functionname'] = 'f(x,y)=x*y on [-'+str(D)+','+str(D)+']^2'
    params['dim_in'] = 2
    params['dim_out'] = 1

    return x_dat, y_dat, test_x_dat, test_y_dat

def cos():
    D = math.pi*2
    a = 4
    dat_size = params['dat_size']

    x_dat = (torch.rand((dat_size,1))*2*D)-D
    y_dat = torch.cos(a*x_dat)

    test_x_dat = torch.linspace(-D, D, steps=params['test_samples']) # [-D,D]
    test_y_dat = torch.cos(a*test_x_dat)

    params['functionname'] = 'f(x)=cos('+str(a)+'x) on [-'+str(D)+','+str(D)+']'
    params['dim_in'] = 1
    params['dim_out'] = 1

    return x_dat, y_dat, test_x_dat, test_y_dat

# currently unusable
def multivar_poly():
    D = 3
    dat_size = params['dat_size']

    x_dat = (torch.rand((dat_size,2))*2*D)-D
    y_dat = (x_dat[:,0]**6)*(x_dat[:,1])-2*(x_dat[:,0]**3)*(x_dat[:,1]**3)-4*(x_dat[:,1]*x_dat[:,0])+3*x_dat[:,0]+9*x_dat[:,0]
    # x_dat[:,0]*x_dat[:,1]

    # test = .....

    params['functionname'] = 'p(x,y)= x^6y-2x^3y^3-4xy+3x+9y on [-'+str(D)+','+str(D)+']^2'
    params['dim_in'] = 2
    params['dim_out'] = 1

    # print(max(y_dat)) #tensor(3602.2234)
    # ...

    return x_dat, y_dat

## Willson Basis
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
    t = t - 0.25 # ! translate g to be active on -0.25 to 0.25 for fair gabor sampling
    return math.sqrt(2)*wills(t+0.25)*willc(t-0.25) #*math.sqrt(2)

def wilsong(l,k,x):
    if(l == 0):
        return willg(x-(k/2))
    if(l+k % 2 == 0):
        return math.sqrt(2)*math.cos(2*math.pi*l*x)*willg(x-(k/2))
    else:
        return math.sqrt(2)*math.sin(2*math.pi*l*x)*willg(x-(k/2))

def MTg(xi,x,t): # modulete xi, translate x
    # M T g
    re = math.cos(2*math.pi*xi*t)*willg(t-x)
    im = math.sin(2*math.pi*xi*t)*willg(t-x)
    return re,im

def willsongenerator(): # predict the actual generator function
    # note: slope is virtually nonexistent for numeric purpose
    dat_size = params['dat_size']

    x_dat = (torch.rand((dat_size,1)))-0.25 # (-0.25,0.75)
    res = np.array(list(map(willg, x_dat.numpy())))
    y_dat = torch.tensor(res,dtype=torch.float).view(-1,1)
    # print(x_dat.shape)
    # print(y_dat.shape)

    test_x_dat = (torch.rand((dat_size,1)))-0.25 # 
    res = np.array(list(map(willg, test_x_dat.numpy())))
    test_y_dat = torch.tensor(res).view(-1,1)

    params['functionname'] = 'f(x)=g(x) willson g on (-0.25,0.75)'
    params['dim_in'] = 1
    params['dim_out'] = 1

    return x_dat, y_dat, test_x_dat, test_y_dat


def getSample():
    alpha = 0.25*0.001
    beta = 1*0.25*0.001
    D = 0.25 # 

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

def approxwilg():
    # aprox normal wilg as below

    dat_size = params['dat_size']

    x_dat = (torch.rand((dat_size,1))*0.5)-0.25 # (-0.25,0.25)
    res = np.array(list(map(willg, x_dat.numpy())))
    y_dat = torch.tensor(res,dtype=torch.float).view(-1,1)

    test_x_dat = torch.linspace(-0.25, 0.25, steps=params['test_samples']) # 
    res = np.array(list(map(willg, test_x_dat.numpy())))

    test_y_dat = torch.tensor(res).view(-1,1)

    params['functionname'] = 'willson g on (-0.25,0.25)'
    params['dim_in'] = 1
    params['dim_out'] = 1

    return x_dat, y_dat, test_x_dat, test_y_dat


def willsonrandomsamplegenerator():
    # generate random sample via gabor sampling of our wilson basis g gen. fun.
    # work on -0.25, 0.25, alpha = ,beta = , D = 0.25, d=1
    # !! sample solely real or imaginary componet with 50%-50% chance
    ## ignore values outside -1/4,1/4

    # MTg(xi,x,t), sample xi and x

    x,xi = getSample()
    f = lambda t : MTg(xi,x,t)
    component = ''

    dat_size = params['dat_size']

    x_dat = (torch.rand((dat_size,1))*0.5)-0.25 # (-0.25,0.25)
    # print('xdat',x_dat)
    res = np.array(list(map(f, x_dat.numpy())))
    if(random.random() > 0.5):
        res = res[:,0] # real aprt
        component = 're'
    else:
        res = res[:,1] # im part
        component = 'im'
    y_dat = torch.tensor(res,dtype=torch.float).view(-1,1)
    # print(x_dat.shape)
    # print(y_dat.shape)

    test_x_dat = torch.linspace(-0.25, 0.25, steps=params['test_samples']) # 
    res = np.array(list(map(f, test_x_dat.numpy())))
    if(component == 're'):
        res = res[:,0]
    else:
        res = res[:,1]

    test_y_dat = torch.tensor(res).view(-1,1)

    params['functionname'] = 'willson g on (-0.25,0.25) with x='+str(x)+',xi='+str(xi)+',component:'+component
    params['dim_in'] = 1
    params['dim_out'] = 1

    return x_dat, y_dat, test_x_dat, test_y_dat

def identity(): # 1d -> 1d
    # approximating the identiy on -D,D
    D = 1
    dat_size = params['dat_size']

    x_dat = (torch.rand((dat_size,1))*2*D)-D
    y_dat = x_dat.clone()

    test_x_dat = torch.linspace(-D, D, steps=params['test_samples']) # [-D,D]
    test_y_dat = test_x_dat.clone()

    params['functionname'] = 'f(x)=x on [-'+str(D)+','+str(D)+']'
    params['dim_in'] = 1
    params['dim_out'] = 1

    return x_dat, y_dat, test_x_dat, test_y_dat

### ==== End Functions ==== 


# static for now
class FunctionDataset(Dataset):
    def __init__(self,x_dat, y_dat):

        self.length = len(x_dat)
        self.y_dat = y_dat #.to(params['device'])
        self.x_dat = x_dat #.to(params['device'])

    def __getitem__(self, index):
        sample = self.x_dat[index]
        label = self.y_dat[index]
        return sample, label

    def __len__(self):
        return self.length

dire = 'experiments/'+str(params['experiment_id'])+'/'

if os.path.exists(dire): # should only happen in main..
    if params['reset_experiment']:
        log.info(("Deleted old experiment of id ",params['experiment_id']))
        shutil.rmtree(dire)

if not os.path.exists(dire):
    os.makedirs(dire)

    log.info(('Creating new Experiment',))

    ### create experiment
    params['seed'] = random.randint(0,100000)
    setSeed(params['seed'])

    # x_dat, y_dat, test_x_dat, test_y_dat  = cos()
    # x_dat, y_dat, test_x_dat, test_y_dat = willsongenerator()
    # x_dat, y_dat, test_x_dat, test_y_dat = identity()
    # x_dat, y_dat, test_x_dat, test_y_dat = data_quadratic()
    # x_dat, y_dat, test_x_dat, test_y_dat = data_product()
    ##x_dat, y_dat = multivar_poly()
    x_dat, y_dat, test_x_dat, test_y_dat = approxwilg()

    dat_size = params['dat_size']
    train_size = int(dat_size*params['train_ratio']) # training/learning/testing
    valid_size = int(dat_size*params['valid_ratio'])
    params['validation_size'] = valid_size
    # train_size = len(data_set) - test_size

    indices = list(range(dat_size))
    train_indices, validation_indices = indices[0:train_size], indices[train_size:train_size+valid_size]

    ## save all
    # params, x_dat, y_dat, test_x_dat, test_y_dat, train_indices, validation_indices
    torch.save(x_dat,dire+'x_dat.pyt')
    torch.save(y_dat,dire+'y_dat.pyt')
    torch.save(test_x_dat,dire+'test_x_dat.pyt')
    torch.save(test_y_dat,dire+'test_y_dat.pyt')

    np.save(dire+'train_indices',train_indices)
    np.save(dire+'validation_indices',validation_indices)
    np.save(dire+'params', params) 

else:
    ### load experiment
    log.info(('Loading previous Experiment',))

    x_dat = torch.load(dire+'x_dat.pyt')
    y_dat = torch.load(dire+'y_dat.pyt')
    test_x_dat = torch.load(dire+'test_x_dat.pyt')
    test_y_dat = torch.load(dire+'test_y_dat.pyt')

    train_indices = list(np.load(dire+'train_indices.npy'))
    validation_indices = list(np.load(dire+'validation_indices.npy'))
    params = np.load(dire+'params.npy').item()
    #load_experiment(params['experiment_id'])
    # read_dictionary = np.load('my_file.npy').item()



### create necessary things
setSeed(params['seed'])

data_set = FunctionDataset(x_dat, y_dat)

training_sampler = SubsetRandomSampler(train_indices)
validation_sampler = SubsetRandomSampler(validation_indices)

# pins memory!
training_loader = DataLoader(data_set, sampler=training_sampler, batch_size=params['batch_size'], shuffle=False, num_workers=params['dl_workers'], pin_memory=True)
validation_loader = DataLoader(data_set, sampler=validation_sampler, batch_size=params['validation_size'], shuffle=False, num_workers=params['dl_workers'], pin_memory=True)
# ^ batch_size = valsize? .. one batch used for plotting and validation_error

log.info(("params:",params))

# Split; Training Validation Test
# Test = linspace

#####

class Net(nn.Module):

    # width = L(phi)
    # num_layers = W(phi)
    def __init__(self,width,num_layers,in_dim, out_dim): 
        super(Net, self).__init__()

        self.layers = []
        self.num_layers = num_layers
        self.width = width
        self.use_leaky = False

        if(params['structure']=='affine'):
            if num_layers == 1:
                self.layers = [nn.Linear(in_dim,out_dim)]
                self.add_module("l0", self.layers[0])
            else:
                for i in range(num_layers):
                    l = False
                    if i == 0:
                        l = nn.Linear(in_dim,width)
                    elif i == num_layers-1:
                        l = nn.Linear(width,out_dim)
                    else:
                        l = nn.Linear(width,width)
                    self.layers.append(l)
                    self.add_module("l"+str(i), l)
        elif(params['structure']=='1dconv'): ## TODO, see params
            kernel_size = 1 # /
            if num_layers == 1:
                self.layers = [nn.Conv1d(in_dim,out_dim,kernel_size)]
                self.add_module("l0", self.layers[0])
            else:
                for i in range(num_layers):
                    l = False
                    if i == 0:
                        l = nn.Conv1d(in_dim,width,kernel_size)
                    elif i == num_layers-1:
                        l = nn.Conv1d(width,out_dim,kernel_size)
                    else:
                        l = nn.Conv1d(width,width,kernel_size)
                    self.layers.append(l)
                    self.add_module("l"+str(i), l)


    def forward(self, x):
        # print(x)

        if params['skip_conn']: # assumes layer depth is even -> 2 ReLU skip blocks
        # first and last layer are for dimensionality, only skip in between those
            if self.use_leaky:
                x =  F.leaky_relu(self.layers[0](x))
            else:
                x =  F.relu(self.layers[0](x))

            for i in range(1,self.num_layers-1,2):
                x_skip = x

                l1 = self.layers[i]
                l2 = self.layers[i+1]

                if self.use_leaky: # apply ReLU twice
                    x = F.leaky_relu(l2(F.leaky_relu(l1(x))))
                else:
                    x = F.relu(l2(F.relu(l1(x)))) # !!

                x = torch.add(x,x_skip) # adding the skip connection to residual

            x = self.layers[self.num_layers-1](x)

        else: # normal DNN, no skip connections
            for i,l in enumerate(self.layers):
                if i  == self.num_layers-1:
                    x = l(x)
                else:
                    if self.use_leaky:
                        x = F.leaky_relu(l(x))
                    else:
                        x = F.relu(l(x)) # !!
                    # TODO
                    #  torch.nn.functional.rrelu(input, lower=1./8, upper=1./3, training=False, inplace=False) to Tensor[source]
                    # https://arxiv.org/pdf/1505.00853.pdf

        return x
        # Add ReLU for 1x1 layer

    def getDead(self,x): # get 0-constant relu's for input x
        # assumes not skip conn!
        # print(len(self.layers))
        result = [] # top layer first
        for i,l in enumerate(self.layers):
            if i  == self.num_layers-1:
                x = l(x)
            else:
                x = F.relu(l(x)) # !!
                tmp = list(x.detach().numpy())
                result.append([h == 0 for h in tmp])
                # result.append(tmp)
        return result


    def getB(self): # max abs weight
        l = []

        for p in self.parameters():
            m = np.max(np.abs(list(p.view(-1)))).item()
            l.append(m)

        return np.max(l)

    def clampWeights(self,min,max):
        def clmp(module,min,max):
            if type(module) == nn.Linear:
            # if hasattr(module, "weight"):
                weights = module.weight.data
                weights.clamp_(min,max)
                bias = module.bias.data
                bias.clamp_(min,max)

        f = lambda m : clmp(m,min,max)
        self.apply(f)


def train(network):

    pid = mp.current_process().pid #.name
    pname = mp.current_process().name
    if not isinstance(network.width, int): # TODO
        net_width = int(network.width.item())
    else:
        net_width = network.width
    if not isinstance(network.width, int):
        net_depth = int(network.num_layers.item())
    else:
        net_depth = network.num_layers

    if(params['cuda']):
        network.cuda(params['device'])

    val_error_list = []
    tra_error_list = []

    min_error = -1
    min_network = 0 # keep a deepcopy of least error network

    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()

    learning_rate = params['lr']
    # decay_var = params['decay_var'] # half each time

    optimizer = optim.SGD(network.parameters(), lr=learning_rate) # SGD
    # optimizer = optim.Adam(network.parameters(),lr=learning_rate)
    # based on infinity norm
    # optimizer = optim.Adamax(network.parameters(),lr=learning_rate)

    # patience 10 -> ?
    LRPlateauScheduler = ReduceLROnPlateau(optimizer, mode='min',factor=0.1, patience=params['lr_plateau_patience'], 
                                verbose=True, threshold=0.0001, threshold_mode='rel', 
                                cooldown=0, min_lr=0, eps=1e-08 )
    # TODO: https://pytorch.org/docs/master/_modules/torch/optim/lr_scheduler.html#ExponentialLR

    if params['use_leaky_to_train']:
        network.use_leaky = True

    dl_iter = iter(training_loader) # next(iter(dl)) is messed up

    i = 0
    while True:
        
        try:
            x_val, target = next(dl_iter)
        except StopIteration:
            dl_iter = iter(training_loader)
            x_val, target = next(dl_iter)

        # log.info(x_val.shape)

        if(params['cuda']):
            x_val = x_val.to(params['device'])
            target = target.to(params['device'])

        optimizer.zero_grad() # don't accumulate gradient, does per default!


        x_val = x_val.view(params['batch_size'],params['dim_in']) # TODO!
        # target = target.view(126,1)
        # out = out.view(-1) #?
        # pass in tensor shaped as N x x_val where N is batch size {!!!}

        out = network(x_val) ## multiprocessing dies here usually, when it does
        
        target = target.view(params['batch_size'],params['dim_out'])

        loss = criterion(out,target) 

        # only for 1 -> 1 dim.
        if(params['zero_error']):
            zero_error = abs(network(torch.tensor([0.0]))).item() # TODO
            loss = loss + (100*zero_error)
        

        valid_error = -1
        if(i % params['validation_intervals'] == 0):
            # Have to calculate before training step to match the train_error
            valid_error = validationError(network) # calculate every epoch? V. Expensive
            LRPlateauScheduler.step(valid_error)


        train_error = loss.item()

        # log.info(("loss crit: ",train_error))

        loss.backward() #backprop
        optimizer.step()


        if params['clamp_weights']:
            network.clampWeights(-params['clamp'],params['clamp'])


        # Stopping criterium
        if params['min_epochs'] <= i:
            if params['max_epochs'] != -1 and params['max_epochs'] <= i:
                break


        if(i % params['validation_intervals'] == 0):
            if min_error == -1:
                min_error = valid_error
                min_network = network

            val_error_list.append(valid_error)
            tra_error_list.append(train_error)


            # TODO less naive patience? running avg?
            patience = params['break_early_patience']
            if len(val_error_list) > patience and params['min_epochs']<=i:

                if min_error < np.min(val_error_list[-patience:])-params['mach_eps']:
                    log.info(('pid:',pid,',',"! Breaking Early Patience Reached, min error:",min_error,"curr err:",valid_error))
                    break 

            #log.info(('pid:',pid,',',i,'/',"tra err:",train_error,"valid err:",valid_error,"min valid err:",min_error))

            # deepcopy minimum error network
            if min_error > valid_error:
                min_error = valid_error
                min_network = copy.deepcopy(network) # ! slightly expensive


        i = i+1



    if params['use_leaky_to_train']:
        network.use_leaky = False
        min_network.use_leaky = False # in case of deepcopy stuff..

    log.info(('pid:',pid,',',"val err list: ",val_error_list))
    log.info(('pid:',pid,',',"tra err list: ",tra_error_list))


    ## ToDo: Plotting inside multipooling locks threads by third party matplotlib or friends ?

    # plt.plot(val_error_list,label='val_error_list',linewidth=1.0)
    # plt.plot(tra_error_list,label='tra_error_list',linewidth=1.0)
    # plt.legend()
    # plt.savefig('plots/single_err_highres-'+pname+'-'+str(net_width)+'-'+str(net_depth)+'.svg', format='svg', dpi=1200)
    # # plt.savefig('single_err.png')
    # # plt.show() # don't interrupt the heatmap
    # plt.close() # plotting library deadlocks? don't plot here?


    return min_network


def validationError(network):
    dl_iter = iter(validation_loader) #
    x_coords, correct_y_coords = next(dl_iter)

    # y_coords = [network(torch.tensor([tmp])).detach().numpy() for tmp in x_coords]

    if(params['cuda']):
        x_coords = x_coords.to(params['device'])
        correct_y_coords = correct_y_coords.to(params['device'])

    x_coords = x_coords.view(params['validation_size'],params['dim_in'])

    out = network(x_coords)
    y_coords = out.view(params['validation_size'],params['dim_out'])

    # TODO, redundant
    if(params['cuda']):
        y_coords = torch.tensor(y_coords, device=params['device']).view(-1)
        correct_y_coords = torch.tensor(correct_y_coords, device = params['device']).view(-1)
    else:
        y_coords = torch.tensor(y_coords).view(-1) # TODO, assumes outdim = 1
        correct_y_coords = torch.tensor(correct_y_coords).view(-1)

    criterion = nn.MSELoss()

    error = criterion(y_coords,correct_y_coords)

    if(params['zero_error']):
            zero_error = abs(network(torch.tensor([0.0]))).item() # TODO
            error = error + (100*zero_error)

    return error.item()


def testingError(network):
    x_coords = test_x_dat.view(-1,params['dim_in'])

    out = network(x_coords)
    y_coords = out.view(-1,params['dim_out'])

    y_coords = torch.tensor(y_coords).view(-1)
    correct_y_coords = torch.tensor(test_y_dat).view(-1)

    inf_norm = np.max(np.abs(y_coords.detach().numpy() -correct_y_coords.detach().numpy() ))

    return inf_norm.item()

    ## infinity norm
    # bla = np.linspace(0,1,error_resolution)
    # x_coords = bla.tolist() #samples
    # correct_y_coords = (bla*bla).tolist()
    # error = torch.max(torch.abs(torch.tensor(y_coords)-torch.tensor(correct_y_coords)))


# plot graph
def plot_nn(network):

    ## TODO: plot based on test data

    if(params['dim_in'] == 1 and params['dim_out'] == 1):
        x_coords, correct_y_coords = next(iter(validation_loader))

        # x_coords = np.linspace(0,1,1000).tolist()

        x_coords = x_coords.view(params['validation_size'],1)
        out = network(x_coords)
        y_coords = out.view(params['validation_size'],1).detach().numpy()
        # y_coords = [network(torch.tensor([tmp])).detach().numpy() for tmp in x_coords] # cleanu detach

        plt.scatter(x_coords,y_coords, marker='.',linewidth=1.0,s=1)
        plt.scatter(x_coords,correct_y_coords, marker='.',linewidth=1.0,s=1)

        plt.savefig('plots/single.svg',format='svg', dpi=1200) # !! slow with so many markers
        # plt.show()
        plt.close()

    ## warning: may be outdated code
    if(params['dim_in'] == 2 and params['dim_out'] == 1):
        x_coords, correct_y_coords = next(iter(validation_loader))

        # x_coords = np.linspace(0,1,1000).tolist()

        x_coords = x_coords.view(params['validation_size'],params['dim_in'])
        out = network(x_coords)
        y_coords = out.view(params['validation_size'],params['dim_out'])
        # log.info(x_coords)
        # log.info(y_coords)
        x = x_coords[:,0].view(-1).detach().numpy()
        y = x_coords[:,1].view(-1).detach().numpy()
        z = y_coords.view(-1).detach().numpy()

        # TODO also plot correct_y_coords~

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        s = ax.plot_trisurf(x, y, z, cmap=plt.cm.jet, linewidth=0.01) # 0.02, cmap=plt.cm.viridis
        fig.colorbar( s, shrink=0.5, aspect=5)
        plt.savefig('plots/single.png')
        # plt.show()
        plt.close()


# sample multiple networks of same toplogy with different random initialized weights
# returns tuple (network_with_least_error, average_error_from_sample)
#@timeout_decorator.timeout(params['optSampleTimeout'], use_signals=False)
@retry(before_sleep=before_sleep_log(log, logging.INFO)) # only for timeout error?
@timeout_decorator.timeout(params['optSampleTimeout']) #, use_signals=False
def optFromSample(samples,width,layers):
    result = False
    least_error = -1
    error_list = []
    err_inf_list = []

    for n in range(samples):
        log.info(("Sampling ",(n+1)," of ",samples))

        net = Net(width,layers,params['dim_in'],params['dim_out'])
        net = train(net)
        # error = testingError(net) # TODO, exclusive?
        error = validationError(net)
        err_inf = testingError(net)

        err_inf_list.append(err_inf)
        error_list.append(error)
        if least_error == -1 or (least_error > error): # use inf norm here?
            log.info(("Found Better Network at error: ",error, "for width:",width,"layers",layers))
            least_error = error
            result = net # no need to deepcopy
        
    # ...
    avg_err = np.average(error_list)
    min_err = np.min(error_list)
    avg_inf_err = np.average(err_inf_list)
    min_inf_err = np.min(err_inf_list)
    return (result,avg_err,min_err,avg_inf_err,min_inf_err)


# Explore the Space
# Heatmap of Accuracy for Depth / Width
def exploreHeatmap():
    # max_W = params['heatmap_w']
    # max_L = params['heatmap_d']

    h = params['heat_h']
    w = params['heat_w']

    # avg_map = np.zeros((max_W,max_L))
    avg_map = np.zeros((len(w),len(h)))
    net_map = {}
    min_map = np.zeros((len(w),len(h)))
    avginf_map = np.zeros((len(w),len(h)))
    mininf_map = np.zeros((len(w),len(h)))

    size_to_sample = params['spp']

    # TODO retire workers == 1
    ## WARNING: workesr == 1 BREAKS IN PLOTTING FOR NOW, still necessary for profiling since can't profile multipooling due to double pickling?
    if(params['workers'] == 1):
        for w in range(0,avg_map.shape[0]):
            for d in range(0,avg_map.shape[1]):
                net,avg_err,min_error,avginf_err,mininf_err = optFromSample(size_to_sample,w+1,d+1)
                avg_map[w][d] = avg_err
                min_map[w][d] = min_error
                avginf_map[w][d] = avginf_err
                mininf_map[w][d] = mininf_err
                log.info(avg_map)
                log.info(min_map)

    else:
        mp.set_start_method('spawn')
        # mp.set_start_method('forkserver')

        # https://github.com/pytorch/pytorch/issues/2245
        # fork() isn't really meant to be used in multi-threaded programs (it's undefined behaviour IIRC)
        # -> but it's the default?! because it's a drop-in for python and blame python?

        # also: https://github.com/pytorch/pytorch/issues/3492

        with Pool(processes=params['workers']) as pool:
            # h = list(range(1,max_L+1))
            # h = [2,4,6,8,10,20]
            # h = [2,4,8] # tiny exp.
            # w = list(range(1,max_W+1))
            # w = [2,6,10,14,16,18,20,24,30]
            # w = [2,5,18,24] # example
            s = params['spp']
            sizes = np.array(list(itertools.product(w, h)))
            args = [np.concatenate( ([s],x)) for x in sizes]
            # better resolution for cos, experiment:
            # manipulate w, h(1-5)
            ## have an args map, then fill in respective 0 
            ## labels from w / h

            p = pool.starmap(optFromSample, args)
            p_tmp = np.array(p)
            log.info((p_tmp,))
            log.info((p_tmp[:,1],)) # avg_error
            for res in p_tmp:
                w_tmp = res[0].width 
                d_tmp = res[0].num_layers
                ## TODO here then... prefill with (h,w), find and replace?

                ### TODO here, find width and height, also fix 0 arrays upstairs there
                w_ind = w.index(w_tmp)
                h_ind = h.index(d_tmp)

                net_map[w_ind,h_ind] = res[0]
                # avg_map[w-1][d-1] = res[1] # avg_error
                avg_map[w_ind][h_ind] = res[1]
                min_map[w_ind][h_ind] = res[2] # min_error
                avginf_map[w_ind][h_ind] = res[3]
                mininf_map[w_ind][h_ind] = res[4]
            # net,avg_err = optFromSample(size_to_sample,w+1,d+1)
    
    # for saving
    hash = str(str(time.time()).split(".")[0])

    log.info(('Saving Results',))
    dire = 'experiments/'+str(params['experiment_id'])+'/'
    # torch.save(p,dire+'resto'+hash)
    # np.save(dire+'resnp'+hash,p_tmp)
    np.save(dire+'resavg'+hash,avg_map) # TODO don't need hash here...
    np.save(dire+'resnet'+hash,net_map) # needs Net class from above defined to load, see pickle
    np.save(dire+'resmin'+hash,min_map)
    np.save(dire+'resavginf'+hash,avginf_map)
    np.save(dire+'resmininf'+hash,mininf_map)
    # save height and width... in params...

    log.info(('Plotting Results',))

    #### Grid of plots of min_l2 err networks
    ### ASSUMES: inn-dim = out-dim = 1
    fig, ax = plt.subplots(len(h),len(w), sharex='col', sharey='row')
    for i in range(len(h)):
        for j in range(len(w)):
            # ax[i, j].text(0.5, 0.5, str((i, j)), fontsize=18, ha='center')
            # x_coords, correct_y_coords = next(iter(validation_loader))

            # x_coords = np.linspace(0,1,1000).tolist()

            # x_coords = x_coords.view(params['validation_size'],1)
            # out = network(x_coords)
            # y_coords = out.view(params['validation_size'],1).detach().numpy()
            # y_coords = [network(torch.tensor([tmp])).detach().numpy() for tmp in x_coords] # cleanu detach
            network = net_map[j,i]

            x_coords, correct_y_coords = next(iter(validation_loader)) # plot valid. set since minimized based on l2 here

            # x_coords = np.linspace(0,1,1000).tolist()

            x_coords = x_coords.view(params['validation_size'],1)
            out = network(x_coords)
            y_coords = out.view(params['validation_size'],1).detach().numpy()
            # y_coords = [network(torch.tensor([tmp])).detach().numpy() for tmp in x_coords] # cleanu detach

            ax[i, j].scatter(x_coords,y_coords, marker='.',linewidth=1.0,s=1)
            ax[i, j].scatter(x_coords,correct_y_coords, marker='.',linewidth=1.0,s=1)

            # x_coords = np.linspace(0,1,100)
            # y_coords = x_coords
            # ax[i, j].scatter(x_coords,y_coords, marker='.',linewidth=1.0,s=1)
            # ax[i, j].scatter(x_coords,correct_y_coords, marker='.',linewidth=1.0,s=1)

    # fig.show()
    # plt.show()
    plt.savefig('plots/plotgridl2min'+hash+'.svg',format='svg', dpi=1200)
    plt.savefig('plots/plotgridl2min'+hash+'.png',format='png', dpi=1200) # easier on pc
    plt.close()

    # print(avg_map)
    ax = sns.heatmap(np.rot90(avg_map), cmap="Blues", annot=True) # , linewidth=0.5
    ax.set(xlabel='Width', ylabel='Layers',)
    # ax.set(xticklabels=list(range(1,max_W+1)),yticklabels=reversed(list(range(1,max_L+1) )))
    ax.set(xticklabels=w,yticklabels=list(reversed(h))) ##
    # label
    ax.text(0,0, str(params), fontsize=6, va='bottom', ha='left', wrap=True)
    ax.set_title('avg error, l2 norm')
    plt.savefig('plots/heatmapavg'+hash+'.svg',format='svg', dpi=1200)
    plt.close()

    ## !TODO remove redundancy... #################

    ax = sns.heatmap(np.rot90(min_map), cmap="Blues", annot=True) # , linewidth=0.5
    ax.set(xlabel='Width', ylabel='Layers',)
    ax.set(xticklabels=w,yticklabels=list(reversed(h))) ##
    # label
    ax.text(0,0, str(params), fontsize=6, va='bottom', ha='left', wrap=True)
    ax.set_title('min error, l2 norm')
    plt.savefig('plots/heatmapmin'+hash+'.svg',format='svg', dpi=1200)
    plt.close()

    ax = sns.heatmap(np.rot90(avginf_map), cmap="Blues", annot=True) # , linewidth=0.5
    ax.set(xlabel='Width', ylabel='Layers',)
    ax.set(xticklabels=w,yticklabels=list(reversed(h))) ##
    # label
    ax.text(0,0, str(params), fontsize=6, va='bottom', ha='left', wrap=True)
    ax.set_title('avg error, inf norm')
    plt.savefig('plots/heatmapavginf'+hash+'.svg',format='svg', dpi=1200)
    plt.close()

    ax = sns.heatmap(np.rot90(mininf_map), cmap="Blues", annot=True) # , linewidth=0.5
    ax.set(xlabel='Width', ylabel='Layers',)
    ax.set(xticklabels=w,yticklabels=list(reversed(h))) ##
    # label
    ax.text(0,0, str(params), fontsize=6, va='bottom', ha='left', wrap=True)
    ax.set_title('min error, inf norm')
    plt.savefig('plots/heatmapmininf'+hash+'.svg',format='svg', dpi=1200)
    plt.close()



def exploreIndividual():
    depth = 5
    width = 4

    # net = Net(8,2,params['dim_in'],params['dim_out'])
    net = Net(width,depth,params['dim_in'],params['dim_out'])
    net = train(net)
    log.info(("validat err: ",validationError(net)))

    log.info(("params:",list(net.parameters())))
    log.info(("B: ",net.getB() ))

    log.info(("depth:",depth))
    log.info(("width:",width))

    plot_nn(net) # ToDo, remove domain



def examineDeads():
    depth = 6
    width = 5

    for i in range(0,100):
        # net = Net(8,2,params['dim_in'],params['dim_out'])
        net = Net(width,depth,params['dim_in'],params['dim_out'])
        net = train(net)
        log.info(("validat err: ",validationError(net)))

        log.info(("params:",list(net.parameters())))
        log.info(("B: ",net.getB() ))

        log.info(("depth:",depth))
        log.info(("width:",width))

        dire = 'experiments/'+str(params['experiment_id'])+'/'
        torch.save(net,dire+'net'+str(i))

        print(i,'of',100)


def countDeads():

    dire = 'experiments/'+str(params['experiment_id'])+'/'

    dead_list = []
    dead_layer_list = []
    for i in range(0,100):
        net = torch.load(dire+'net'+str(i))
        log.info(('net '+str(i),))
        log.info((list(net.parameters()),))
        # r = net.getDead(torch.tensor([1.0]))
        # print('bbb',torch.tensor([1.0]))
        res = []
        for i in test_x_dat:
            # print('aaa',i.view(-1))
            r = net.getDead(i.view(-1))
            res.append(r)
        
        # print(np.mean(res)) # % of activity of all ReLU's
        activity_per_relu = np.mean(res,axis=0)
        log.info((activity_per_relu,)) # gives percentage of activity per ReLU of single network
        # numb_dead = np.array(activity_per_relu).flatten().
        numb_dead = np.count_nonzero( activity_per_relu == 0 ) # amazing
        log.info((numb_dead,))
        dead_list.append(numb_dead)
        dpl = np.count_nonzero( activity_per_relu == 0 , axis = 1)
        log.info((dpl,)) # dead per layer
        dead_layer_list.append(dpl)

    log.info((dead_layer_list,))
    log.info((dead_list,))

    log.info(('avg dead: ',np.mean(dead_list)))
    log.info(('avg dead per layer: ',np.mean(dead_layer_list, axis = 0)))
    # todo: average this over all trained networks, make some helpful graphic
    totalrelusperlayer = 4
    totalrelus = 4*totalrelusperlayer

    log.info(('percent dead: ',100*(np.mean(dead_list)/totalrelus)))
    log.info(('pecent dead per layer: ',100*(np.mean(dead_layer_list, axis = 0)/totalrelusperlayer)))


def graphFreqs():
    toplot = {}

    for ide in range(500,506):

        dire = 'experiments/'+str(ide)+'/'
        ps = np.load(dire+'params.npy')[()]
        nets = np.load(dire+'resnet.npy')
        # print(ps)
        tmp = ps['functionname']
        tmp= tmp.replace('f(x)=cos(','')
        tmp=tmp.replace('x) on [-6.283185307179586,6.283185307179586]','')
        print('aaa',tmp)
        freq = int(tmp)

        for i in range(0,6):
            net = nets[()][(0,i)] #...
            toplot[(freq,net.num_layers)] = net
         # amazing...

    print(toplot)

    fig, ax = plt.subplots(7,12)#, sharex='col', sharey='row')
    for ii,i in enumerate(range(1,7)):
        for ij,j in enumerate(range(2,14,2)):
            # load testx dat and y dat!

            network = toplot[(i,j)]

            # x_coords, correct_y_coords = next(iter(validation_loader))
            x_coords = torch.linspace(- math.pi*2, math.pi*2, steps=10000)

            x_coords = x_coords.view(10000,1)
            out = network(x_coords)
            y_coords = out.view(10000,1).detach().numpy()

            ax[ii, ij].scatter(x_coords,y_coords, marker='.',linewidth=1.0,s=1)
            # ax[i, j].scatter(x_coords,correct_y_coords, marker='.',linewidth=1.0,s=1)

    # fig.show()
    # plt.show()
    # plt.savefig('happyplot.svg',format='svg', dpi=1200)
    plt.savefig('happyplot.png',format='png', dpi=1200) # easier on pc
    plt.close()

@retry(before_sleep=before_sleep_log(log, logging.INFO))
@timeout_decorator.timeout(params['optSampleTimeout'])
def trainrandomwilson(expid):
    depth = 4 # 8, 12
    width = 6 # TODO

    x_dat, y_dat, test_x_dat, test_y_dat = willsonrandomsamplegenerator()
    log.info((expid,(params['functionname'])))

    params['seed'] = int(time.time())
    setSeed(params['seed'])

    dat_size = params['dat_size']
    train_size = int(dat_size*params['train_ratio'])
    valid_size = int(dat_size*params['valid_ratio'])
    params['validation_size'] = valid_size

    indices = list(range(dat_size))
    train_indices, validation_indices = indices[0:train_size], indices[train_size:train_size+valid_size]

    # save
    dire = 'experiments/'+str(params['experiment_id'])+'/'

    torch.save(x_dat,dire+'x_dat'+str(expid)+'.pyt')
    torch.save(y_dat,dire+'y_dat'+str(expid)+'.pyt')
    torch.save(test_x_dat,dire+'test_x_dat'+str(expid)+'.pyt')
    torch.save(test_y_dat,dire+'test_y_dat'+str(expid)+'.pyt')

    np.save(dire+'train_indices'+str(expid),train_indices)
    np.save(dire+'validation_indices'+str(expid),validation_indices)
    np.save(dire+'params'+str(expid), params) 

    # create validation stuff
    setSeed(params['seed'])
    data_set = FunctionDataset(x_dat, y_dat)

    training_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)

    training_loader = DataLoader(data_set, sampler=training_sampler, batch_size=params['batch_size'], shuffle=False, num_workers=params['dl_workers'], pin_memory=True)
    validation_loader = DataLoader(data_set, sampler=validation_sampler, batch_size=params['validation_size'], shuffle=False, num_workers=params['dl_workers'], pin_memory=True)

    log.info(("params:",params))
    
    # do training and save results
    samples = params['spp']
    result,avg_err,min_err,avg_inf_err,min_inf_err = optFromSample(samples,width,depth)
    np.save(dire+'network'+str(expid),result)
    stuff = (avg_err,min_err,avg_inf_err,min_inf_err) 
    np.save(dire+'networkstats'+str(expid),stuff)

    # call many times...



def errorCorrelateDeads():
    dire = 'experiments/'+str(params['experiment_id'])+'/'

    dead_list = []
    error_list = []
    val_err_list = []
    dead_layer_list = []
    for i in range(0,100):
        net = torch.load(dire+'net'+str(i))
        log.info(('net '+str(i),))
        log.info((list(net.parameters()),))
        # r = net.getDead(torch.tensor([1.0]))
        # print('bbb',torch.tensor([1.0]))
        res = []
        for i in test_x_dat:
            # print('aaa',i.view(-1))
            r = net.getDead(i.view(-1))
            res.append(r)
        
        # print(np.mean(res)) # % of activity of all ReLU's
        activity_per_relu = np.mean(res,axis=0)
        log.info((activity_per_relu,)) # gives percentage of activity per ReLU of single network
        # numb_dead = np.array(activity_per_relu).flatten().
        numb_dead = np.count_nonzero( activity_per_relu == 0 ) # amazing
        log.info((numb_dead,))
        dead_list.append(numb_dead)
        dpl = np.count_nonzero( activity_per_relu == 0 , axis = 1)
        log.info((dpl,)) # dead per layer
        dead_layer_list.append(dpl)

        error = testingError(net)
        val_error = validationError(net)
        error_list.append(error)
        val_err_list.append(val_error)

    log.info((dead_layer_list,))
    log.info((dead_list,))

    log.info(('avg dead: ',np.mean(dead_list)))
    log.info(('avg dead per layer: ',np.mean(dead_layer_list, axis = 0)))
    # todo: average this over all trained networks, make some helpful graphic
    totalrelusperlayer = 4
    totalrelus = 4*totalrelusperlayer

    log.info(('percent dead: ',100*(np.mean(dead_list)/totalrelus)))
    log.info(('pecent dead per layer: ',100*(np.mean(dead_layer_list, axis = 0)/totalrelusperlayer)))

    np.save(dire+'dead_list',dead_list)
    np.save(dire+'error_list',error_list)
    np.save(dire+'val_error_list',val_err_list)
    np.save(dire+'dead_layer_list',dead_layer_list)
    


def examinewilsons():
    # x = torch.load('experiments/800/test_x_dat1.pyt')
    # y = torch.load('experiments/800/test_y_dat1.pyt')

    # plt.scatter(x,y, marker='.',linewidth=1.0,s=1)
    # plt.show()

    # net = np.load('experiments/800/network1.npy')
    # print(list(net[()].parameters()))
    pass

def collectsampleapproxes():
    mp.set_start_method('spawn')

    with Pool(processes=params['workers']) as pool:
        expids = np.array(list(range(1,101)))
        p = pool.map(trainrandomwilson, expids)
    

def plotGaborSamples():
    # 10 spp
    expid = 802
    dire = 'experiments/'+str(expid)+'/'
    # print(params)
    xlist = []
    xilist = []
    complist = []
    errlist = []
    for i in range(1,101):
        p = np.load(dire+'params'+str(i)+'.npy')[()]
        stats = np.load(dire+'networkstats'+str(i)+'.npy')[()]
        print(p['functionname'])
        x = p['functionname'].split(',xi')[0].split('x=')[1]
        x = float(x)
        xi = p['functionname'].split(',xi=')[1].split(',comp')[0]
        xi = float(xi)
        component = p['functionname'].split('nent:')[1]
        component = (component == 're') 
        print(x,xi,component) # component = true if real
        print(stats)
        xlist.append(x)
        xilist.append(xi)
        complist.append(component)
        errlist.append(stats[0])
        # stats = (avg_err,min_err,avg_inf_err,min_inf_err) 

    baseline = np.load('experiments/805/resavg1546796395.npy')[()][0]
    print('base',['{:.15}'.format(float(x)) for x in baseline])

    print(errlist)

    plt.show()
    print('mean err:','{:.15}'.format(float(np.mean(errlist))))


def main():
    # pass
    # exploreIndividual()
    exploreHeatmap()
    # examineDeads()
    # countDeads()
    # graphFreqs()
    # collectsampleapproxes()
    # examinewilsons()
    # errorCorrelateDeads()
    # plotGaborSamples()



# mp spawn protection
if __name__ == '__main__':
    main()


