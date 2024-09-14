from typing import NamedTuple, Optional, Dict, Any
from absl import flags
import random
import numpy as np

flags.DEFINE_float('epsilon', 0.0, 'Regularization strength for the Sinkhorn loss and the noise added to the data'
                   '(which will be sqrt(epsilon/2) unless passed as a flag.')
flags.DEFINE_integer('n_epochs', 300, 'The number of epochs', lower_bound=1)
flags.DEFINE_integer('batch_size', 200, 'The batch size', lower_bound=1)
flags.DEFINE_integer('n_sinkhorn_steps', 100, 'The number of Sinkhorn steps for each generator iteration',
                     lower_bound=1)
flags.DEFINE_enum('sinkhorn_method', 'sinkhorn_log', ['sinkhorn_log', 'emd'], 'The sinkhorn method to be used')
flags.DEFINE_float('lr', 0.005, 'Adam learning rate.')
flags.DEFINE_integer('val_freq', 5, 'validation loss will be calculated every val_freq steps',
                     lower_bound=1)
flags.DEFINE_integer('dump_freq', 10, 'The number of Sinkhorn steps for each generator iteration',
                     lower_bound=1)
flags.DEFINE_integer('latent_dim', 2, 'Latent dimension',
                     lower_bound=2)
flags.DEFINE_integer('n_train_samples', 60000, 'The number of training samples')
flags.DEFINE_boolean('debias_sinkhorn', False, 'If passed, the sinkhorn divergence will be debiased.')
flags.DEFINE_string('load_path', None, 'The path to load network and optimizer from')
flags.DEFINE_enum('cost_fn', 'l2', ['l1', 'l2'], 'The cost function of the wasserstein distance')
flags.DEFINE_enum('noise', 'normal', ['normal', 'laplace'], 'The noise to be added to the train data')
flags.DEFINE_enum('transfer_fct', 'softplus', ['softplus', 'relu','leakyrelu'], 'The activation of hidden layers')
flags.DEFINE_enum('model', 'perceptron', ['perceptron', 'dcgan'], 'The generator model')
flags.DEFINE_float('data_noise', None, 'The data noise to the model, if Not specified the noise '
                   'compatible with the regularization is added')
flags.DEFINE_float('norm_clip', None, 'The parameter to clip the norm of images/their transform before adding noise'
                   '(the norm clipped is the same as cost)')
# flags.DEFINE_boolean('clip_dft', False, 'If specified, the dft norm is clipped and the loss is calculated in the dft')
flags.DEFINE_boolean('use_proj', False, 'If specified, the train data/its transform will be projected onto a lower-dimensional space')
flags.DEFINE_integer('proj_dim', 28*7, 'Projection dimension')
flags.DEFINE_boolean('denoise_wavelet', False, 'If true, wavelet denoising will be done on each image instead'
                     'of adding entropic regularization')
# flags.DEFINE_boolean('use_wavelet', False, 'If true, wavelet transform will be applied to the train and generated data')
flags.DEFINE_float('coef_thresh', None, 'The quantile below which to set the transform coefficients to 0 (only if '
                   'data_transform is not none, thresh = 1 sets all the coefficients to 0 but the largest one, thresh=0'
                  'leaves the coefficients untouched')
flags.DEFINE_enum('data_transform', None, ['dct', 'wavelet'], 'The transform to apply to the data and the generator')
flags.DEFINE_boolean('approx_norm', False, 'If true, each transform will be approximated to clip_norm in l2 distance')
flags.DEFINE_float('mean_eps', 0.0, 'privacy budget to calculate the mean for norm approximation')
flags.DEFINE_boolean('use_bn', False, 'whether to use batch normalization in the Generator')



class Opt(NamedTuple):
    exp_id: int = int(f'{random.random():.16f}'[2:])
    n_epochs: int = 300
    batch_size: int = 200
    # entropic regularization strength (for consistency with gaussian denoising has to be sigma=sqrt(eps/2))
    epsilon: float = 0.0
    n_sinkhorn_steps: int = 100 #100 # L in Genevay's work
    lr: float = 0.005
    latent_dim: int = 2
    img_size = (28, 28, 1)
    network_architecture: Dict[str, Any] = dict(
        n_hidden_gener_1=500, # 1st layer decoder neurons
        n_hidden_gener_2=500, # 2nd layer decoder neurons
        n_input=np.prod(img_size), # MNIST data input (img shape: 28*28)
        n_z=latent_dim  # dimensionality of latent space
    )
    sinkhorn_method: str = 'sinkhorn_log' #method to compute the sinkhorn step
    n_train_samples: Optional[int] = None # if None, all train samples will be used
    n_test_samples: Optional[int] = None # if None, all test samples will be used
    val_freq: int = 5 # calculate validation loss every val_loss_freq epochs 
    dump_freq: int = 10 # save progress (model weights and history) every dump_freq epochs
    n_display: int = 20 #number of images to display in a grid
    debias_sinkhorn: bool = False # note that in parameters if the debias is not present, it is false and that also indicates that the loss is taken to be 1/N * current loss
    debias_grad_sinkhorn: bool = False #if this is specified, the batch size is increased 2-fold
    load_path: str = None
    align_noise_mean: bool = False
    cost_fn: str = 'l2'
    noise: str = 'normal'
    data_noise: float = np.sqrt(epsilon/2) if noise == 'normal' else epsilon# sigma
    transfer_fct: str = 'softplus'
    model: str = 'perceptron'
    norm_clip: Optional[float] = None
#     clip_dft: bool = False
    use_proj: bool = False
    proj_dim: int = 28*7
    denoise_wavelet: bool = False
    data_transform: Optional[str] = None
    coef_thresh: Optional[float] = None
    mean_eps: float = 0.
    approx_norm: bool = False
    use_bn: bool = False
        
def create_training_params_from_dict(d):
    d_default = Opt._asdict(Opt())
    d_default.update({key: val for key, val in d.items() if key in Opt.__annotations__})
    if d_default['data_noise'] is None:
        d_default['data_noise'] = np.sqrt(d_default['epsilon']/2) if d_default['noise'] == 'normal' else d_default['epsilon']
    return Opt(**d_default)
        
