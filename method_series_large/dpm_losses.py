import torch
import numpy as np
import torch.nn.functional as F
import math
from torch.autograd import Variable
from datetime import datetime
from method_series.utils import cosine_beta_schedule, CELoss, sample_discrete_features, sample_discrete_feature_noise, compute_batched_over0_posterior_distribution
from method_series.utils import PredefinedNoiseScheduleDiscrete, DiscreteUniformTransition, DiscreteMarginalTransition
from torch_geometric.loader import NeighborSampler, NeighborLoader
from torch_geometric.data import Data
from tqdm import tqdm

class label_mask_diffusion_model(torch.nn.Module):
    def __init__(self, device, timesteps):
        super(label_mask_diffusion_model, self).__init__()

        #neglegible_threshold = 1e-5
        #delta = 1.0 - neglegible_threshold ** (1/timesteps)
        #cum_one_minus_delta = (1.0 - delta) ** torch.arange(0, timesteps+1) # 0, 1, 2, ..., T
        #one_minus_cum_one_minus_delta = 1.0 - cum_one_minus_delta
        #delta_mul_cum_one_minus_delta = delta * cum_one_minus_delta

        #self.register("cum_one_minus_delta", cum_one_minus_delta.to(device[0]))
        #self.register("one_minus_cum_one_minus_delta", one_minus_cum_one_minus_delta.to(device[0]))
        #self.register("delta_mul_cum_one_minus_delta", delta_mul_cum_one_minus_delta.to(device[0]))
        #print(self.cum_one_minus_delta, self.one_minus_cum_one_minus_delta, self.delta_mul_cum_one_minus_delta)
        #input()

        betas = cosine_beta_schedule(timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        alphas_cumprod_prev = torch.cat(
            (torch.tensor([1], dtype=torch.float64), alphas_cumprod[:-1]), 0
        )
        posterior_variance = betas
        self.register("betas", betas.to(device[0]))
        self.register("alphas", alphas.to(device[0]))
        self.register("alphas_cumprod", alphas_cumprod.to(device[0]))
        self.register("sqrt_alphas", torch.sqrt(alphas).to(device[0]))
        self.register("alphas_cumprod_prev", alphas_cumprod_prev.to(device[0]))
        self.register("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).to(device[0]))
        self.register("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod).to(device[0]))
        self.register("thresh", (1-self.alphas)/self.sqrt_one_minus_alphas_cumprod)
        self.register("log_one_minus_alphas_cumprod", torch.log(1 - alphas_cumprod).to(device[0]))
        self.register("posterior_variance", posterior_variance.to(device[0]))

        #print(self.betas)
        #print(self.alphas_cumprod)
        #print(1 - self.alphas_cumprod)
        p = torch.cat([torch.Tensor([1.0]).to(self.betas.device)] + [(self.betas[i] * self.alphas_cumprod[i-1] / (1.0 - self.alphas_cumprod[i])).reshape((1,)) for i in range(1, timesteps+1)])
        self.register("lambda2", p)
        #print(self.lambda2)
        #print(self.lambda2 * 2700 * (1 - self.alphas_cumprod))
        #input()

        self.num_timesteps = timesteps
        self.device = device

    def register(self, name, tensor):
        self.register_buffer(name, tensor.type(torch.float32))

    def q_sample(self, x, t):
        p = self.alphas_cumprod[t].repeat(x.shape[:-1] + (1,))
        noise = 1 - torch.bernoulli(p)
        # 1 mask, 0 keep
        return x * (1-noise), noise.squeeze(1)


class diffusion_model(torch.nn.Module):
    def __init__(self, device, timesteps):
        super(diffusion_model, self).__init__()

        betas = cosine_beta_schedule(timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        alphas_cumprod_prev = torch.cat(
            (torch.tensor([1], dtype=torch.float64), alphas_cumprod[:-1]), 0
        )
        posterior_variance = betas
        self.register("betas", betas.to(device[0]))
        self.register("alphas", alphas.to(device[0]))
        self.register("alphas_cumprod", alphas_cumprod.to(device[0]))
        self.register("sqrt_alphas", torch.sqrt(alphas).to(device[0]))
        self.register("alphas_cumprod_prev", alphas_cumprod_prev.to(device[0]))
        self.register("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).to(device[0]))
        self.register("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod).to(device[0]))
        self.register("thresh", (1-self.alphas)/self.sqrt_one_minus_alphas_cumprod)
        self.register("log_one_minus_alphas_cumprod", torch.log(1 - alphas_cumprod).to(device[0]))
        self.register("posterior_variance", posterior_variance.to(device[0]))
        self.num_timesteps = timesteps
        self.device = device

    def register(self, name, tensor):
        self.register_buffer(name, tensor.type(torch.float32))

    def q_sample(self, x, t):
        noise = torch.randn_like(x)
        return (
            self.sqrt_alphas_cumprod[t] * x
            + self.sqrt_one_minus_alphas_cumprod[t] * noise, noise
        )


class mllp_losses_large(object):

    def __init__(self, num_timesteps, device):
        self.diff_Y = label_mask_diffusion_model(device=device, timesteps = num_timesteps)
        self.num_timesteps = num_timesteps
        self.device = device

    # Loss function
    def loss_fn(self, model, x, adj, y, train_mask, batch = 1, wtts=False, emphasize_labeled=False, learn_identity=False):
        losses = .0
        picked_timesteps = []
        criterion = torch.nn.CrossEntropyLoss(reduction='sum' if wtts else 'mean')


        for i in range(0, batch):
            if 2*i < batch: # assume batch is even
                t = self.sample_time(self.device)
                picked_timesteps.append(t)
            else:
                t = self.num_timesteps - picked_timesteps[i % len(picked_timesteps)] + 1
            q_Y_sample, noise = self.diff_Y.q_sample(y, t)
            pred_y = model(x, q_Y_sample, adj, t, self.num_timesteps)

            # Compute losses for observed nodes/home/jhs/DPM-GSP-semi-supervised/models
            # and we should avoid label leakage
            if wtts:
                if emphasize_labeled:
                    if learn_identity:
                        num_considered = noise.size(0)
                        cur_train_mask = train_mask
                        cur_non_train_mask = ~train_mask
                    else:
                        num_considered = noise.sum()
                        cur_train_mask = torch.logical_and(train_mask, noise.bool())
                        cur_non_train_mask = torch.logical_and(~train_mask, noise.bool())


                    losses = losses + self.diff_Y.lambda2[t] * num_considered / (2.0 * cur_train_mask.sum()) * criterion(
                        pred_y[cur_train_mask], y[cur_train_mask]
                    )
                    losses = losses + self.diff_Y.lambda2[t] * num_considered / (2.0 * cur_non_train_mask.sum()) * criterion(
                        pred_y[cur_non_train_mask], y[cur_non_train_mask]
                    )
                else:
                    if learn_identity: # theoretically, not that sound...
                        losses = losses + self.diff_Y.lambda2[t] * criterion(
                            pred_y, y
                        )
                    else: # theoretically sound solution
                        losses = losses + self.diff_Y.lambda2[t] * criterion(
                            pred_y[noise.bool()], y[noise.bool()]
                        )
            else:
                if emphasize_labeled:
                    if learn_identity:
                        cur_train_mask = train_mask # label leakage is OK as we are not learning identity mapping for all nodes
                        cur_non_train_mask = ~train_mask # label leakage is OK as we are not learning identity mapping for all nodes
                    else:
                        cur_train_mask = torch.logical_and(train_mask, noise.bool())
                        cur_non_train_mask = torch.logical_and(~train_mask, noise.bool())

                    losses = losses + criterion(
                        pred_y[cur_train_mask], y[cur_train_mask]
                    )
                    losses = losses + criterion(
                        pred_y[cur_non_train_mask], y[cur_non_train_mask]
                    )
                else:
                    if learn_identity:
                        losses = losses + criterion(
                            pred_y, y
                        )
                    else:
                        losses = losses + criterion(
                            pred_y[noise.bool()], y[noise.bool()]
                        )

        return losses/batch

    def estimate(self, model, x, adj, y, mask, temp=0.02, device=None):
        updated_y = torch.zeros_like(y)
        masked_flag = torch.ones_like(mask)

        subgraph_loader = NeighborSampler(adj, sizes=[-1], num_nodes=x.size(0),
                                              batch_size=4096, shuffle=False,
                                              num_workers=12)

        for i in reversed(range(1, self.num_timesteps+1)):
            y0_hat_logits = model.inference(x, updated_y, subgraph_loader, torch.tensor([i]).to(device), self.num_timesteps, device)
            y0_hat_probs = torch.softmax(y0_hat_logits / temp, dim=-1)
            y0_hat = torch.multinomial(y0_hat_probs, num_samples=1) # of shape (|V|, 1)

            p = self.diff_Y.lambda2[i].repeat(masked_flag.shape)
            should_denoised = torch.logical_and(masked_flag, torch.bernoulli(p) if i > 1 else torch.ones_like(masked_flag)) # ensure all are denoised. is this necessary?
            updated_y = updated_y + should_denoised.unsqueeze(-1) * F.one_hot(y0_hat.squeeze(1), updated_y.shape[1]).float()
            masked_flag[should_denoised] = 0
        return F.one_hot(torch.argmax(updated_y, dim = -1), updated_y.shape[1]).float()

    # Manifold-constrained sampling
    def mc_estimate(self, model, x, adj, y, mask, temp = 0.02, coef = 1, corr_dist='Unif', device=None):
        model.eval()
        with torch.no_grad():
            updated_y = torch.zeros_like(y)
            masked_flag = torch.ones_like(mask)
            # masked_flag[num_central_nodes:] = 0

            #mistake = torch.zeros_like(mask)
            data = Data(x=x, y=updated_y, edge_index=adj)

            for i in tqdm(reversed(range(1, self.num_timesteps+1))):

                #predicted_y = torch.argmax(y0_hat_logits, dim = -1)
                #gt_y = torch.argmax(y, dim = -1)
                #corrected = (predicted_y == gt_y)
                #print("=====================================> {}".format(i))
                #print(corrected.sum().item(), corrected.shape[0], float(corrected.sum().item()) / float(corrected.shape[0]))
                #print(corrected[masked_flag].sum().item(), masked_flag.sum().item(), float(corrected[masked_flag].sum().item()) / float(masked_flag.sum().item() + 1e-6))
                #print(corrected[mistake].sum().item(), mistake.sum().item(), float(corrected[mistake].sum().item()) / float(mistake.sum().item() + 1e-6))

                y_hat = torch.zeros_like(y)

                noisy_flag = masked_flag # so do NOT modify noisy_flag in-inplace
                p = self.diff_Y.lambda2[i].repeat(noisy_flag.shape)
                num_should_denoised = (p * noisy_flag).sum() # expected number of noisy nodes to be denoised
                should_denoised = torch.zeros_like(noisy_flag)

                if corr_dist.startswith('LF'):
                    remaining_labeled = torch.logical_and(noisy_flag, mask)
                    num_remaining_labeled = remaining_labeled.sum().long()
                    num_labeled_to_sample = min(torch.ceil(num_should_denoised).long(), num_remaining_labeled)
                    if num_labeled_to_sample > 0:
                        remaining_labeled_indices = remaining_labeled.nonzero().squeeze(-1)
                        picked_labeled_indices = remaining_labeled_indices[torch.randperm(remaining_labeled_indices.size(0))[:num_labeled_to_sample]]
                        should_denoised[picked_labeled_indices] = 1
                        noisy_flag = torch.logical_and(noisy_flag, ~mask)
                        num_should_denoised -= num_labeled_to_sample
                        #p = (num_should_denoised / noisy_flag.sum()).repeat(noisy_flag.shape)

                if num_should_denoised > 0:
                    remaining_indices = noisy_flag.nonzero().squeeze(-1)
                    if corr_dist.endswith('Unif'):
                        picked_indices = remaining_indices[torch.randperm(remaining_indices.size(0))[:int(num_should_denoised.item())]] if i > 1 else remaining_indices # ensure all are denoised. is this necessary?
                    else:
                        raise ValueError(corr_dist)

                    should_denoised[picked_indices] = 1

                    # from torch_geometric.utils import k_hop_subgraph
                    # sub_nodes, sub_edge_index, _, sub_mask = k_hop_subgraph(
                    #     picked_indices, num_hops=2, edge_index=adj, relabel_nodes=True
                    # )
                    # subgraph_loader = NeighborSampler(sub_edge_index, sizes=[-1],
                    #                                   batch_size=4096, shuffle=False,
                    #                                   num_workers=12)

                    import time
                    t_loader = time.time()
                    loader = NeighborLoader(data, num_neighbors=[-1, -1], input_nodes=picked_indices, batch_size=512, shuffle=False)
                    print(f"time to instantial a loader: {time.time()-t_loader :.5f}")

                    y_hat = torch.zeros_like(y)
                    # y_hat_picked_logits = model.inference(x[sub_nodes], updated_y[sub_nodes], subgraph_loader, torch.tensor([i]).to(device), self.num_timesteps, device)
                    y_hat_picked_logits = model.inference(loader, torch.tensor([i]).to(device), self.num_timesteps, device)
                    y_hat_picked_probs = torch.softmax(y_hat_picked_logits / temp, dim=-1)
                    y_hat_picked = torch.multinomial(y_hat_picked_probs, num_samples=1) # of shape (|V|, 1)
                    y_hat_picked = F.one_hot(y_hat_picked.squeeze(1), updated_y.shape[1]).float()
                    y_hat[picked_indices] = y_hat_picked

                #print(torch.logical_and(should_denoised, mask).sum(), should_denoised.sum())

                if i > 1:
                    should_refine = torch.logical_and(should_denoised, mask)
                    y_hat[should_refine] = y[should_refine]

                updated_y = updated_y + should_denoised.unsqueeze(-1) * y_hat
                masked_flag[should_denoised] = 0

                #mistake = torch.logical_or(mistake, torch.logical_and(~corrected, torch.logical_and(should_denoised, ~mask)))

            return F.one_hot(torch.argmax(updated_y, dim = -1), updated_y.shape[1]).float()

    def sample_time(self, device):
        t = torch.randint(1, self.num_timesteps+1, (1,), device=device[0]).long()
        return t



