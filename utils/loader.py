import torch
import random
import numpy as np
from models.model import Denoising_Model, OurDenoising_Model, Simple_Model, LPA_Model, P_Model, CLGNN_Model, G3NN_Model, Simple_Res_Model, Denoising_Model_Large, Simple_Model_Large
from method_series.dpm_losses import continuous_dpm_losses, discrete_dpm_losses, mllp_losses
from method_series_large.dpm_losses import mllp_losses_large
from method_series.dpm_losses import simple_losses
import torch.nn.functional as F
import os

def load_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def load_device():
    if torch.cuda.is_available():
        device = list(range(torch.cuda.device_count()))
    else:
        device = 'cpu'
    return device


def load_model(params):
    params_ = params.copy()
    type = params_['alpha']
    params_.pop('alpha')

    if type == 0:
        model = Denoising_Model(**params_)
    else:
        model = OurDenoising_Model(**params_)
    return model

def load_model_large(params):
    params_ = params.copy()
    model = Denoising_Model_Large(**params_)
    return model


def load_p_model(params):
    params_ = params.copy()
    params_.pop('alpha')
    model = P_Model(**params_)
    return model


def load_simple_model(params):
    params_ = params.copy()
    model = Simple_Model(**params_)
    return model

def load_simple_res_model(params):
    params_ = params.copy()
    model = Simple_Res_Model(**params_)
    return model


def load_simple_model_large(params):
    params_ = params.copy()
    model = Simple_Model_Large(**params_)
    return model

def load_lpa_model(params):
    params_ = params.copy()
    params_.pop('alpha')
    model = LPA_Model(**params_)
    return model


def load_clgnn_model(params):
    params_ = params.copy()
    params_.pop('alpha')
    model = CLGNN_Model(**params_)
    return model


def load_g3nn_model(params):
    params_ = params.copy()
    params_.pop('alpha')
    params_.pop('model')
    params_.pop('num_layers')
    params_.pop('num_linears')
    model = G3NN_Model(**params_)
    return model


######################################################



def load_model_optimizer(params, config_train, device):
    model = load_model(params)
    model = model.to(f'cuda:{device[0]}')

    optimizer = torch.optim.Adam(model.parameters(), lr=config_train.lr_diffusion,
                                    weight_decay=config_train.weight_decay_diffusion)
    scheduler = None
    if config_train.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config_train.lr_decay)
    return model, optimizer, scheduler

def load_model_optimizer_large(params, config_train, device):
    model = load_model_large(params)
    model = model.to(f'cuda:{device[0]}')

    optimizer = torch.optim.Adam(model.parameters(), lr=config_train.lr_diffusion,
                                    weight_decay=config_train.weight_decay_diffusion)
    scheduler = None
    if config_train.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config_train.lr_decay)
    return model, optimizer, scheduler


def load_p_model_optimizer(params, config_train, device, gmnn = False):
    model = load_p_model(params)
    model = model.to(f'cuda:{device[0]}')

    if gmnn:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config_train.lr,
                                        weight_decay=config_train.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config_train.lr,
                                        weight_decay=config_train.weight_decay)

    scheduler = None
    if config_train.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config_train.lr_decay)
    return model, optimizer, scheduler


def load_lpa_model_optimizer(params, config_train, device):
    model = load_lpa_model(params)
    model = model.to(f'cuda:{device}')

    optimizer = torch.optim.Adam(model.parameters(), lr=config_train.lr,
                                    weight_decay=config_train.weight_decay)

    scheduler = None
    if config_train.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config_train.lr_decay)

    return model, optimizer, scheduler


def load_g3nn_model_optimizer(params, config_train, device):
    model = load_simple_model(params)
    generator = load_g3nn_model(params)
    model = model.to(f'cuda:{device}')
    generator = generator.to(f'cuda:{device}')

    optimizer = torch.optim.Adam([{'params':model.parameters()},{'params':generator.parameters()}], lr=config_train.lr,
                                        weight_decay=config_train.weight_decay)

    scheduler = None
    if config_train.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config_train.lr_decay)
    return model, generator, optimizer, scheduler


def load_clgnn_model_optimizer(params, config_train, device):
    model = load_clgnn_model(params)
    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
        model = model.to(f'cuda:{device[0]}')


    optimizer = torch.optim.Adam(model.parameters(), lr=config_train.lr,
                                        weight_decay=config_train.weight_decay)

    scheduler = None
    if config_train.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config_train.lr_decay)
    return model, optimizer, scheduler


def load_simple_model_optimizer(params, config_train, device, gmnn = False):

    if params['model'] == 'GATsepConv':
        model = load_simple_res_model(params)

    else:
        model = load_simple_model(params)

    model = model.to(f'cuda:{device[0]}')

    if gmnn:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config_train.lr,
                                        weight_decay=config_train.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config_train.lr,
                                        weight_decay=config_train.weight_decay)

    scheduler = None
    if config_train.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config_train.lr_decay)
    return model, optimizer, scheduler


def load_simple_model_optimizer_large(params, config_train, device, gmnn = False):

    model = load_simple_model_large(params)

    model = model.to(f'cuda:{device[0]}')

    if gmnn:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config_train.lr,
                                        weight_decay=config_train.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config_train.lr,
                                        weight_decay=config_train.weight_decay)

    scheduler = None
    if config_train.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config_train.lr_decay)
    return model, optimizer, scheduler


######################################################


def load_data(config):
    from utils.data_loader import dataloader
    return dataloader(config)


def load_loss_fn(config, device, y=None):
    if config.diffusion.method == 'Continuous':
        return continuous_dpm_losses(config.diffusion.step, device = device)
    elif config.diffusion.method == 'Discrete-Uniform':
        return discrete_dpm_losses(config.diffusion.step, config.data.nlabel, None, config.diffusion.noise_schedule, device = device)
    elif config.diffusion.method == 'Discrete-Marginal':
        y = y.to(torch.long)
        cnt = torch.sum(y, dim=0)
        marginal = cnt / y.size(0)
        return discrete_dpm_losses(config.diffusion.step, config.data.nlabel, marginal, config.diffusion.noise_schedule, device = device)
    elif config.diffusion.method == 'LP':
        if config.data.data in ['ogbn-products']:
            return mllp_losses_large(config.diffusion.step, device = device)

        else:
            return mllp_losses(config.diffusion.step, device = device)


def load_simple_loss_fn(config, device):
    return simple_losses(device = device)


def load_model_params(config):
    config_m = config.model
    nlabel = config.data.nlabel
    # params_ = {'model':config_m.model, 'num_linears': config_m.num_linears, 'nhid': config_m.nhid,
    #                 'nfeat': config.data.nfeat, 'nlabel': nlabel,'num_layers':config_m.num_layers, 'alpha': 0.1}

    params_ = {'model':config_m.model, 'num_linears': config_m.num_linears, 'nhid': config_m.nhid,
                    'nfeat': config.data.nfeat, 'nlabel': nlabel,'num_layers':config_m.num_layers, 'alpha': config_m.type}
    return params_


def load_model_params_large(config):
    config_m = config.model
    nlabel = config.data.nlabel
    params_ = {'num_linears': config_m.num_linears, 'nhid': config_m.nhid,
               'nfeat': config.data.nfeat, 'nlabel': nlabel,'num_layers':config_m.num_layers, 'dropout': 0.5}

    return params_

