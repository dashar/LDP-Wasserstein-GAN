import numpy as np
import torch
from torch import nn
import torchvision
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import ot
import time
import pickle as pkl
import random
from typing import NamedTuple, Optional, Dict, Any
from tqdm import tqdm
import pywt
import scipy
from scipy.stats import norm
import cvxpy as cp

import sys, os

class Generator_DCGAN(nn.Module):
    def __init__(self, latent_dim=100, use_bn=True, activation='leakyrelu'):
        super().__init__()
        ngf = 64
        nz = 100
        nc = 1
        self.latent_dim = latent_dim
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf*4, 3, 2, 0, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # state size. (ngf*4) x 3 x 3
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 0, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(ngf*2, ngf, 3, 2, 0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d(ngf, nc, 3, 2, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 28 x 28
        )

    def forward(self, input):
        return self.main(input).view(-1, 28*28)
    #     self.latent_dim = nz = latent_dim
    #     self.use_bn = use_bn
    #     if activation == 'relu':
    #         self.activation = nn.functional.relu
    #     elif activation == 'leakyrelu':
    #         self.activation = lambda x: nn.functional.leaky_relu(x, negative_slope=0.2)
    #     else:
    #         raise ValueError(f"unknown activation function {activation}")
        
    #     self.fc = nn.Linear(nz, 256*7*7)
    #     self.trans_conv1 = nn.ConvTranspose2d(in_channels=256,
    #                                           out_channels=128,
    #                                           kernel_size=3,
    #                                           stride=2,
    #                                           padding=1,
    #                                           output_padding=1)
    #     self.trans_conv1_bn = nn.BatchNorm2d(128)
    #     self.trans_conv2 = nn.ConvTranspose2d(in_channels=128, 
    #                                           out_channels=64,
    #                                           kernel_size=3, 
    #                                           stride=1,
    #                                           padding=1)
    #     self.trans_conv2_bn = nn.BatchNorm2d(64)
    #     self.trans_conv3 = nn.ConvTranspose2d(in_channels=64,
    #                                           out_channels=32, 
    #                                           kernel_size=3,
    #                                           stride=1,
    #                                           padding=1)
    #     self.trans_conv3_bn = nn.BatchNorm2d(32)
    #     self.trans_conv4 = nn.ConvTranspose2d(in_channels=32,
    #                                           out_channels=1,
    #                                           kernel_size=3,
    #                                           stride=2,
    #                                           padding=1,
    #                                           output_padding=1)

    # def forward(self, x):
    #     x = self.fc(x)
    #     x = x.view(-1, 256, 7, 7)
    #     x = nn.functional.relu(self.trans_conv1(x))
    #     x = self.trans_conv1_bn(x) if self.use_bn else x
    #     x = nn.functional.relu(self.trans_conv2(x))
    #     x = self.trans_conv2_bn(x) if self.use_bn else x
    #     x = nn.functional.relu(self.trans_conv3(x))
    #     x = self.trans_conv3_bn(x) if self.use_bn else x
    #     x = self.trans_conv4(x)
    #     x = torch.tanh(x).view(-1, 28*28)
    #     return x
    
class MyDataset():
    # Class to use instead of pytorch torch.utils.data.Dataset for training with a projection matrrix
    # if from batches the list of batches is passed as data_source
    def __init__(self, data_source, batch_size, from_batches=False):
        # Always drop the last batch
        self.data_source = data_source
        self.batch_size = batch_size
        self.n_batches = len(self.data_source) // batch_size if not from_batches else len(data_source)
#         self.batched_idxs = permutation[:self.n_batches*batch_size].reshape(self.n_batches, batch_size)
#         self.process_fn = process_fn
        self.from_batches = from_batches
#         self.batched_idxs = list(map(list, list(batched_idxs)))

    def __iter__(self):
        if self.from_batches:
            batch_permutation = np.random.permutation(self.n_batches)
        else:
            # need to shuffle between the batches too, so we create new batching
            permutation = np.random.permutation(len(self.data_source))
            self.batched_idxs = permutation[:self.n_batches*self.batch_size].reshape(self.n_batches, self.batch_size)
            batch_permutation = np.arange(self.n_batches).astype(int)
        for idx in batch_permutation:
            if self.from_batches:
                batch = self.data_source[idx]
            else:
                batch = torch.stack([self.data_source[i] for i in self.batched_idxs[idx]], axis = 0)
#             batch = self.process_fn(batch, idx) if self.process_fn is not None else batch
            yield batch
            

    def __len__(self) -> int:
        return self.n_batches
    
    
def cost_mat(X,Y,dist='l2', max_size = 400):
    def cost_mat_unbatched(X, Y):
        if dist == 'l2':
            XX = torch.square(X).sum(1).view(-1, 1)
            YY = torch.square(Y).sum(1).view(1, -1)
            XY = torch.matmul(X, Y.T)
            C = XX + YY - 2*XY;
        elif dist == 'l1':
            diff = X.unsqueeze(1) - Y.unsqueeze(0)

            C = (torch.sign(diff)*diff).sum(-1)
        else:
            raise ValueError("Unknown distance")
        return C
    if max_size is None:
        return cost_mat_unbatched(X, Y)
    else:
        n = X.shape[0]
        m = Y.shape[0]
        X_tr, last_batch_X = X[:n - n%max_size], X[n - n%max_size:]
        Y_tr, last_batch_Y = Y[:m - m%max_size], Y[m - m%max_size:]
        C = torch.empty((n, m), device = X.device).float()
        X_tr, last_batch_X = X[:n - n%max_size], X[n - n%max_size:]
        Y_tr, last_batch_Y = Y[:n - n%max_size], Y[n - n%max_size:]
        X_tr = X_tr.view(-1, max_size, *X.shape[1:])
        Y_tr = Y_tr.view(-1, max_size, *Y.shape[1:])

        for i in range(X_tr.shape[0]):
            for j in range(Y_tr.shape[0]):
                C[i*max_size:(i+1)*max_size, j*max_size:(j+1)*max_size] = cost_mat_unbatched(X_tr[i], Y_tr[j])
                if last_batch_X.shape:
                    C[X_tr.shape[0]*max_size:, j*max_size:(j+1)*max_size] = cost_mat_unbatched(last_batch_X, Y_tr[j])
            if last_batch_Y.shape:
                C[i*max_size:(i+1)*max_size, Y_tr.shape[0]*max_size:] = cost_mat_unbatched(X_tr[i], last_batch_Y)
        C[X_tr.shape[0]*max_size:, Y_tr.shape[0]*max_size:] = cost_mat_unbatched(last_batch_X, last_batch_Y)
    return C

def sinkhorn_loss(X,Y, method, reg, debiased=False, dist='l2', n_sinkhorn_steps=100, batch_size=200):
    '''
    method used for the solver either 'sinkhorn','sinkhorn_log',
        'sinkhorn_stabilized', see those function for specific parameters
    '''
    if method == 'emd':
        ot_fn = lambda C: ot.lp.emd2(torch.ones(C.shape[0]).to(C.device)/C.shape[0],
                                  torch.ones(C.shape[1]).to(C.device)/C.shape[1],
                                  C,
                                  numItermax=n_sinkhorn_steps)
    else:
        ot_fn = lambda C: ot.bregman.sinkhorn2(torch.ones(C.shape[0]).to(C.device)/C.shape[0],
                                               torch.ones(C.shape[1]).to(C.device)/C.shape[1],
                                               C,
                                               method=method,
                                               reg=reg,
                                               batchSize=batch_size,
                                               numItermax=n_sinkhorn_steps)
    if debiased:
        C = cost_mat(X,Y,dist)
        Cxx = cost_mat(X,X,dist)
        Cyy = cost_mat(Y,Y,dist)
        ot_xy = ot_fn(C)
        ot_xx = ot_fn(Cxx)
        ot_yy = ot_fn(Cyy)
        final_cost = ot_xy - (ot_xx + ot_yy)/2
    else:
        C = cost_mat(X,Y,dist)
        ot_xy = ot_fn(C)
        final_cost = ot_xy
    return final_cost 

def get_coeff_array(Yl, Yh):
    coef_arr_tens = Yl
    for y in Yh[::-1]:
        r, c = coef_arr_tens.shape[-2:]
        r1, c1 = y.shape[-2:]
        if r1<r or c1<c:
            y = torch.nn.functional.pad(y, ((0,1))*2)
        batch_channel_shape = coef_arr_tens.shape[:2]
        coef_arr_tens = coef_arr_tens.unsqueeze(2)
        coef_arr_tens = torch.concatenate((coef_arr_tens, y), dim=2
                                         ).reshape(*batch_channel_shape,2, 2*r, c
                                                  ).transpose(-3,-2).reshape(*batch_channel_shape, 2*r,2*c)
        coef_arr_tens = coef_arr_tens[:,:,:r+r1,:c+c1]
    return coef_arr_tens

def get_dct_matrix_transposed(dim):
    I = np.eye(dim)
    dct_matrix_1d_T = np.vstack([scipy.fftpack.dct(I[i], norm = 'ortho') for i in range(dim)])
    dct_matrix_T = np.kron(dct_matrix_1d_T, dct_matrix_1d_T)
    return torch.tensor(dct_matrix_T).float()

def add_noise_tensor_transform(img, noise='normal', noise_scale=0., clip=True):
    assert isinstance(img, torch.Tensor)
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)
    if noise=='normal':
        out = img + noise_scale * torch.randn_like(img)
    elif noise == 'laplace':
        noise = torch.distributions.Laplace(0, 1).sample(sample_shape=(img.shape))
        out = img + noise_scale * noise
    if out.dtype != dtype:
        out = out.to(dtype)
    out = torch.clip(out, -1, 1) if clip else out
    return out

def get_loaders(batch_size, noise='normal', noise_scale=0., use_dct=False, n_proj=None):
    if use_dct:
        dct_matrix_T = get_dct_matrix_transposed(28)
        transform_matrix = dct_matrix_T
    else:
        transform_matrix = torch.eye(28*28)
    transform_matrix *=2 # multiply by two to rescale to [-1,1]
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.LinearTransformation(transform_matrix, 0.5*torch.ones(28*28)),
            lambda img: add_noise_tensor_transform(img, noise, noise_scale, not use_dct),
        ])
    transform_no_noise = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.LinearTransformation(transform_matrix, 0.5*torch.ones(28*28)),
            lambda img: add_noise_tensor_transform(img, noise, 0.0, not use_dct),
        ])
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform),
                       batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform_no_noise),
                       batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader, test_loader

def generate_rnd_matrix(size, proj_dim):
    # Generates a random matrix U of size size x proj_dim
    # U^TU = I
    rnd_state = np.random.get_state()
    mat = scipy.stats.ortho_group.rvs(size)[:,:proj_dim]
    return mat, rnd_state

def get_proj_mat_from_idx_state(idx, state_list):
    rnd_state = np.random.get_state()
    np.random.set_state(state_list[idx])
    proj_mat = generate_rnd_matrix(np.prod(batch.shape[1:]), proj_dim)[0]
    np.random.set_state(rnd_state)
    return proj_mat

def get_proj_mat_from_list(idx, mat_list):
    return mat_list[idx]

def clip_norm(batch, clip_norm_to, norm_p, approx=False, avg=None):
    if clip_norm_to is not None:
        if not approx:
            if norm_p == 1:
                norms = (torch.sign(batch)*batch).view(batch.shape[0], -1).sum(-1, keepdim=True)
            else:
                norms = torch.linalg.norm(batch.view(batch.shape[0], -1), dim=-1, ord=norm_p).view(-1,1)
            batch = batch * torch.clip(clip_norm_to/norms, max=1)
        else:
            print("Approximating the input")
            out = []
            try:
                with open('/content/drive/MyDrive/norm_approximated.pkl', 'rb') as f:
                    batch = pkl.load(f)
            except:
                for i in range(batch.shape[0]):
                    b = batch[i].reshape(-1)
                    x = cp.Variable(batch.shape[-1])
                    if avg is not None:
                        cost = cp.norm(x + avg - b, p=2)
                    else:
                        cost = cp.norm(x - b, p=1)
                    prob = cp.Problem(cp.Minimize(cost), [cp.norm(x, p=norm_p)<=clip_norm_to])
                    prob.solve()
                    out.append(x.value)
                    if (i+1) % 1000 == 0:
                        print(i, 'samples done')
                batch = np.stack(out, axis=0)
                with open('/content/drive/MyDrive/norm_approximated.pkl', 'wb') as f:
                    pkl.dump(batch, f)
    return batch

def project_fn(batch, idx, get_proj_mat_from_idx, noise, noise_scale, clip_norm_to=None, norm_p=1):
    # projects the batch onto a matrix generated from random state in state_list and 
    # adds noise
    proj_mat = get_proj_mat_from_idx(idx)
    batch_projected = torch.matmul(batch.view(batch.shape[0], -1).float(), torch.Tensor(proj_mat).float())
    if clip_norm_to is not None:
        if norm_p == 1:
            norms = (torch.sign(batch_projected)*batch_projected).view(batch_projected.shape[0], -1).sum(-1, keepdim=True)
        else:
            norms = torch.linalg.norm(batch_projected.view(batch_projected.shape[0], -1), dim=-1, ord=norm_p).view(-1,1)
        batch_projected = batch_projected * torch.clip(clip_norm_to/norms, max=1)
    # incorrect -- new noise added every time
    batch_noisy = add_noise_tensor_transform(batch_projected, noise, noise_scale, clip=False)
    return (batch_noisy, proj_mat)

def calculate_mean_privatly(data, eps = 1., p=0.5, maxnorm=28):
    data_norm1 = np.concatenate([data/maxnorm, 
                                 np.sqrt(1 - np.square(data).sum(-1, keepdims=True)/(maxnorm**2))], axis = -1)
    true_avg = data.mean(0)/maxnorm
    d = data_norm1.shape[-1]
    sigma = 1/np.sqrt(d)
    avg = 0
    q_frac = np.exp(eps) / p * (1-p)
    q = q_frac / (q_frac+1)
    for i in range(data_norm1.shape[0]):
        v = data_norm1[i]
        gamma = sigma * norm.ppf(q)
        Z = np.random.uniform() < p
        U= np.random.normal()*sigma
        while (Z and (U<gamma)) or ((not Z) and (U>= gamma)):
            U= np.random.normal()*sigma
        alpha = U
        V_perp = np.random.normal(size = (d,))*sigma
        V_perp -= v.dot(V_perp) * v
        V = alpha*v + V_perp
        m = sigma * norm.pdf(gamma/sigma) * (p/(1-q) - (1-p)/q)
        avg = avg * (i/(i+1)) + V/m * (1/(i+1))
    print('Mean estimation error: ', np.linalg.norm(avg[:-1] - true_avg)/np.linalg.norm(true_avg))
    return avg

def denoise_fn(batch, sigma=None, thresh=None, wt='db1', level=3, noise='laplace'):
    lap_thresh = {0.1: 0.2, 0.2: 0.15, 0.3: 0.11, 0.4: 0.09, 0.5: 0.07,
                  0.6: 0.06, 0.7: 0.05, 0.8: 0.04, 0.9: 0.04, 1.0: 0.04,
                  1.1: 0.04, 1.2: 0.04, 1.3: 0.04, 1.4: 0.04, 1.5: 0.04,
                  1.6: 0.05, 1.7: 0.05, 1.8: 0.06, 1.9: 0.06, 2.0: 0.06,
                  2.1: 0.06, 2.2: 0.06, 2.3: 0.07, 2.4: 0.06, 2.5: 0.07,
                  2.6: 0.08, 2.7: 0.09, 2.8: 0.1, 2.9: 0.1, 3.0: 0.1,
                  3.1: 0.11, 3.2: 0.1, 3.3: 0.12, 3.4: 0.1, 3.5: 0.11,
                  3.6: 0.11, 3.7: 0.11, 3.8: 0.12, 3.9: 0.11, 4.0: 0.14,
                  4.1: 0.11, 4.2: 0.12, 4.3: 0.13, 4.4: 0.12, 4.5: 0.13,
                  4.6: 0.13, 4.7: 0.1, 4.8: 0.14, 4.9: 0.15, 5.0: 0.12}
    gauss_thresh = {0.5: 0.0, 1.0: 0.04, 1.5: 0.04, 2.0: 0.04, 2.5: 0.02, 3.0: 0.02}
    if thresh is None and sigma is not None:
        if noise == 'laplace':
            thresh = lap_thresh[sigma]
        elif noise == 'normal':
            thresh = gauss_thresh[sigma]
        else:
            raise ValueError('Unknown noise')
    elif thresh is None and sigma is None:
        raise ValueError('Cannot handle both thresh and sigma none')
    batch = np.array(batch)
    n_img = batch.shape[0]
    wavelets = [pywt.wavedec2(batch[i].reshape(28,28), wavelet='db1', level=3)
                for i in range(n_img)]
    coeff_array, coeff_slices = list(zip(*list(map(lambda x: pywt.coeffs_to_array(x), wavelets))))
    coeff_array = np.stack(coeff_array, axis = 0)
    coeff_array_abs_sorted = np.sort(np.abs(coeff_array.reshape(n_img,-1)), axis = -1)
    max_idx = -np.ceil(thresh*coeff_array_abs_sorted.shape[1]).astype(int)
    tresh_per_img = coeff_array_abs_sorted[:, max_idx].reshape(n_img, 1, 1)
    filt = coeff_array * (np.abs(coeff_array)>=tresh_per_img)
    img_rec = [pywt.waverec2(pywt.array_to_coeffs(filt[i], coeff_slices[i],output_format='wavedec2'),
                                wavelet=wt) for i in range(batch.shape[0])]
    img_rec = np.clip(img_rec, -1, 1)
    img_rec_all = np.stack(img_rec, axis = 0).reshape(batch.shape[0], -1)
    return torch.tensor(img_rec_all, requires_grad=True)

def get_dataset(train=True, n_data=None):
    # creates MNIST dataset Tensor, the data is rescaled to [-1,1] and converted with 2dim DCT if use_dct
    data = torchvision.datasets.MNIST('./data', train=train, download=True).data
    if n_data is not None:
        idxs = np.random.choice(data.shape[0], n_data)
        data = data[idxs]
    # rescale data to [-1, 1]
    data = 2.0*data/255.0 - 1
    return data

def train(G, optim, loss_fn, train_dl, test_dl=None, val_loss_fn=None, epochs = 25, n_display = 25,
         random_noise_generator=lambda x: torch.rand(x, 100), device='cpu', val_freq=5, dump_freq=10, model_name='tmp'):
    tqdm._instances.clear()
    losses = []
    val_losses = []
    gen_input_size = G.latent_dim
    #Having a fixed sample to monitor the progress of the generator
    sample_size = n_display*n_display
    fixed_samples_img = random_noise_generator(n_display**2)
    fixed_samples_img = fixed_samples_img.to(device)
    if val_loss_fn is None:
        val_loss_fn = loss_fn
    
    #Going into training mode
    G = G.to(device)
    G.train()
    
    for epoch in range(epochs):
        gen_loss_total = 0
        val_loss_total = 0
        gen_out = 0
        for i, train_x in tqdm(enumerate(train_dl), total=len(train_dl)):
#                 print(i)
            # drop the label if present
            if type(train_x) == tuple or type(train_x) == list:
                train_x = [x.to(device) for x in train_x]
            else:
                train_x = train_x.to(device)     #Passing to GPU

          
            gen_in = random_noise_generator(train_x.shape[0])
            gen_in = gen_in.to(device)   #Passing to GPU

            #Generator training
            optim.zero_grad()


            gen_out = G(gen_in.float())     #Feeding noise into the generator

            gen_loss = loss_fn(gen_out, train_x)  #Generator loss calculation
            gen_loss_total += gen_loss.detach().cpu().data / len(train_dl)
            gen_loss.backward()
            optim.step()

        losses.append(gen_loss_total)
        
        if (epoch+1) % val_freq == 0:
            if test_dl is not None:
                G.eval()                    #Going into eval mode to get sample images   
                val_loss_total = 0
                for test_x in test_dl:
                    U = None
                    if type(test_x) == tuple or type(test_x) == list:
                        test_x, U = test_x
                    test_x = test_x.view(test_x.shape[0], -1).to(device)
                    test_samples = random_noise_generator(test_x.shape[0])
                    test_samples = test_samples.to(device)
                    samples = G(test_samples.float())
                    val_loss = val_loss_fn(samples, test_x).detach().cpu().data
                    val_loss_total += val_loss / len(test_dl)
                val_losses.append((epoch, val_loss_total))
                G.train()                   #Going back into train mode
        #Plotting samples every 5 epochs
        if (epoch+1)%dump_freq == 0:
            with open(f'/content/drive/MyDrive//hists/{model_name}.pkl', 'wb') as f:
                pkl.dump({'loss': losses, 'val_loss': val_losses}, f)
            torch.save({
                'model_state_dict': G.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
            }, f'/content/drive/MyDrive//models/{model_name}.pt')
            G.eval()                    #Going into eval mode to get sample images        
            Gz = G(fixed_samples_img.float()).detach().cpu().data
            G.train()                   #Going back into train mode
            canvas = np.transpose(Gz.reshape((n_display, n_display, 28, 28, 1)),
                               axes=[0, 2, 1, 3, 4]).reshape(n_display*28, n_display*28, 1)
            plt.figure(figsize=(10, 10))        
            plt.imshow((1 - canvas)/2, origin="upper", cmap="gray")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'/content/drive/MyDrive/imgs/{model_name}.png')
#             plt.savefig(f'imgs/{opt.sinkhorn_method}_{opt.n_epochs}epochs_{opt.exp_id}.png')
            plt.close()
        
        #Printing losses every epoch
        print(f'Epoch {epoch+1}: Loss = {gen_loss_total}', 
              f'Val Loss = {val_loss_total}' if (epoch+1) % val_freq == 0 and test_dl is not None else '')    
    
    return {'loss': losses, 'val_loss': val_losses}