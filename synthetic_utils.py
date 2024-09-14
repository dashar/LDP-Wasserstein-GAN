import numpy as np
import matplotlib.pyplot as plt

from scipy.special import erfc
from random import choices
from imageio.v2 import imread, imsave
import torch

import geomloss
import pykeops
from pykeops.torch import generic_sum
import pickle as pkl

import sys
sys.path.append('.')

import time


from scipy.spatial.transform import Rotation as R
from typing import NamedTuple, Optional, Dict, Any
from absl import flags
import random
import numpy as np



flags.DEFINE_string('shape_name', 'halfcircle', 'Shape name.')
flags.DEFINE_integer('N', 10000, 'The number of target distribution points')
flags.DEFINE_integer('n_iter', 500, 'The number of training iterations')
flags.DEFINE_float('epsilon', 5, 'The privacy budget')
flags.DEFINE_enum('cost', 'l2', ['l2', 'l1'], 'The cost fn')
flags.DEFINE_float('lr', 0.0001, 'RMSprop learning rate.')
flags.DEFINE_float('scaling', 0.999, 'scaling parameter -- how precise the loss is calculated, 0.5=fast & imprecise -> 1 is slow and precise')
flags.DEFINE_integer('hidden_neurons', 400, 'The number of hidden neurons')
flags.DEFINE_integer('hidden_layers', 2, 'The number of hidden layers')
flags.DEFINE_boolean('use_bn', False, 'Whether to use batch normalization')
flags.DEFINE_string('activation', 'relu', 'The activation fn to use')
flags.DEFINE_enum('input_dist', 'uniform', ['uniform', 'normal'], 'The input distribution to the generator')
flags.DEFINE_list('input_scale', [2, 2], 'The vector of size 2 -- input scale of generator input')
flags.DEFINE_list('input_mean', [-1, -1], 'The vector of size 2 -- input mean of generator input')
flags.DEFINE_string('load_path', None, 'Load path')


def str_list_to_float_list(l):
    return list(map(float, l))

class Opt(NamedTuple):
    exp_id: int = int(f'{random.random():.16f}'[2:])
    shape_name: str = 'halfcircle'
    N: int = 10000
    eps: float = 5
    cost: str = 'l2'
    n_iter: int = 500
    scaling: float = .999
    lr: float = 1e-4
    use_bn: bool = False
    hidden_neurons: int = 400
    hidden_layers: int = 2
    activation: str = 'relu'
    input_scale: list = [2, 2]
    input_mean: list = [-1, -1]
    input_dist: str = 'uniform'
    load_path: Optional[str] = None
        
def create_training_params_from_dict(d):
    d_default = Opt._asdict(Opt())
    d_default.update({key: val for key, val in d.items() if key in Opt.__annotations__})
    d_default['input_mean'] = str_list_to_float_list(d_default['input_mean'])
    d_default['input_scale'] = str_list_to_float_list(d_default['input_scale'])
    return Opt(**d_default)
        

class Generator(torch.nn.Module):
    def __init__(self, use_bn = False, hidden_neurons=400, hidden_layers=2, activation = 'relu'):
        super().__init__()
        self.fc_input = torch.nn.Linear(2, hidden_neurons)
        self.bn_input = torch.nn.BatchNorm1d(hidden_neurons)
        self.use_bn = use_bn
        self.hidden_layers = []
        self.bn_layers = []
        for _ in range(hidden_layers - 1):
            self.hidden_layers.append(torch.nn.Linear(hidden_neurons, hidden_neurons))
            self.bn_layers.append(torch.nn.BatchNorm1d(hidden_neurons))
        self.hidden_layers.append(torch.nn.Linear(hidden_neurons, 2))
        self.hidden_layers = torch.nn.ModuleList(self.hidden_layers)
        self.bn_layers = torch.nn.ModuleList(self.bn_layers)
        
        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'tanh':
            self.activation = torch.nn.Tanh()
        elif activation == 'elu':
            self.activation = torch.nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = torch.nn.LeakyReLU()
        elif activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
#         self.activation_list = [self.activation() for _ in range(hidden_layers)]

    def forward(self, x):
        output = self.fc_input(x)
        output = self.bn_input(output) if self.use_bn else output
        for i, l in enumerate(self.hidden_layers):
            output = self.activation(output)
            output = l(output)
            if self.use_bn and i < len(self.bn_layers):
                output = self.bn_layers[i](output)
        return output

    

def load_image(fname):
    img = imread(fname, mode='L')  # Grayscale
    img = (img[::-1, :]) / 255.0
    return 1 - img

def dilate_img(img):
    im = np.array(img, dtype=np.float32)
    kernel = np.ones((3,3)).astype(np.float32)
    im_tensor = torch.Tensor(np.expand_dims(np.expand_dims(im, 0), 0)) # size:(1, 1, 5, 5)
    kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(kernel, 0), 0)) # size: (1, 1, 3, 3)
    torch_result = torch.clamp(torch.nn.functional.conv2d(im_tensor, kernel_tensor, padding=(1, 1)), 0, 1)
    return torch_result[0,0].numpy()

def erode_img(img):
    im = np.array(img, dtype=np.float32)
    kernel = np.ones((3,3)).astype(np.float32)
    im_tensor = torch.Tensor(np.expand_dims(np.expand_dims(im, 0), 0)) # size:(1, 1, 5, 5)
    kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(kernel, 0), 0)) # size: (1, 1, 3, 3)
    torch_result = 1 - torch.clamp(torch.nn.functional.conv2d(1 - im_tensor, kernel_tensor, padding=(1, 1)), 0, 1)
    return torch_result[0,0].numpy()

def get_sigma(eps, delta, sensitivity):
    def err_f(a):
        return erfc(a)-np.exp(eps)*erfc(np.sqrt(a*a+eps)) - 2*delta
    a_low = -1e3
    a_high = 1e3
    a_mid = (a_low+a_high)/2
    err = err_f(a_mid)
    while np.abs(err) > 1e-5:
        a_low = a_mid if err > 0 else a_low
        a_high = a_mid if err < 0 else a_high
        a_mid = (a_low+a_high)/2
        err = err_f(a_mid)
    a = a_mid
#     sigma_high = sensitivity/np.sqrt(2*eps) + sensitivity/eps * np.sqrt(-2*np.log(2*delta))
    sigma = (a + np.sqrt(a*a+eps)) * sensitivity / (eps*np.sqrt(2))
    return sigma



def make_circle(mesh_size):
    vals = np.stack([np.repeat(np.linspace(-1, 1, mesh_size), mesh_size),
                     np.tile(np.linspace(-1, 1, mesh_size), mesh_size)]).T
    vals = vals[np.linalg.norm(vals, axis = -1) <= 1]
    sensitivity = 2
    return vals, sensitivity

def make_halfcircle(mesh_size):
    vals = np.stack([np.repeat(np.linspace(-1, 1, mesh_size), mesh_size),
                     np.tile(np.linspace(-1, 1, mesh_size), mesh_size)]).T
    vals = vals[np.linalg.norm(vals, axis = -1) <= 1]
    vals = vals[vals[:,0]>=0]
    sensitivity = 2
    return vals, sensitivity

def make_ellipsis(mesh_size, a=1, b=1, rot=0):
    vals = np.stack([np.repeat(np.linspace(-1, 1, mesh_size), mesh_size),
                     np.tile(np.linspace(-1, 1, mesh_size), mesh_size)]).T
    vals = vals[(vals[:,0]/a)**2 + (vals[:,1]/b)**2 <= 1]
    rot_mat = R.from_euler('z', rot, degrees=True).as_matrix()[:-1, :-1]
    vals = vals.dot(rot_mat)
    sensitivity = 2*max(a, b)
    return vals, sensitivity

def make_square(mesh_size):
    vals = np.stack([np.repeat(np.linspace(-1, 1, mesh_size), mesh_size),
                     np.tile(np.linspace(-1, 1, mesh_size), mesh_size)]).T
    sensitivity = 2*np.sqrt(2)
    return vals, sensitivity

def make_rect(mesh_size, a=1, b=1, rot=0):
    vals = np.stack([np.repeat(np.linspace(-1, 1, mesh_size), mesh_size),
                     np.tile(np.linspace(-1, 1, mesh_size), mesh_size)]).T
    vals = vals[np.abs(vals[:,0]) <= a]
    vals = vals[np.abs(vals[:,1]) <= b]
    rot_mat = R.from_euler('z', rot, degrees=True).as_matrix()[:-1, :-1]
    vals = vals.dot(rot_mat)
    sensitivity = 2*np.sqrt(a**2 + b**2)
    return vals, sensitivity

def make_shape(img_path="./density_b_simplified.png"):
    A = load_image(img_path)
    A = dilate_img(dilate_img(A))
    A = A.T
    A[:,:] = A[:,::-1]
    idxs = np.where(A == 1)
    A = A[np.min(idxs[0]):np.max(idxs[0]), :]
    A = A[:, np.min(idxs[1]):np.max(idxs[1])]
    vals = 2*(np.array(list(zip(*np.where(A > 0)))) / (np.array(A.shape)[None,:]-1))-1
    sensitivity = 2*np.sqrt(2)
    return vals, sensitivity

def make_shape_fn(shape_name, n_target_pts_sqrt=201, a=1, b=0.3, rot=30):
    n = n_target_pts_sqrt
    if shape_name == 'circle': 
        return (lambda: make_circle(n))
    elif shape_name == 'square': 
        return (lambda: make_square(n))
    elif shape_name == 'halfcircle':
        return (lambda: make_halfcircle(n))
    elif shape_name == 'shape': 
        return (lambda: make_shape())
    elif shape_name == 'rectangle': 
        return (lambda: make_rect(n, a = 1, b=0.3, rot = 30))
    elif shape_name == 'ellipsis':
        return (lambda: make_ellipsis(n, a = 1, b=0.3, rot = 30))
    else:
        raise ValueError(f'Unknown shape %s' % shape_name)

        
