import torch
import numpy as np
import torch.nn.functional as F
import math
from torch.autograd import Variable
from datetime import datetime
from method_series.utils import cosine_beta_schedule, CELoss, sample_discrete_features, sample_discrete_feature_noise, compute_batched_over0_posterior_distribution
from method_series.utils import PredefinedNoiseScheduleDiscrete, DiscreteUniformTransition, DiscreteMarginalTransition


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


class mllp_losses:
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
            pred_y = model(x, q_Y_sample, adj, t, self.num_timesteps, train=True)

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

    def estimate(self, model, x, adj, y, mask, temp=0.02):
        updated_y = torch.zeros_like(y)
        masked_flag = torch.ones_like(mask)
        for i in reversed(range(1, self.num_timesteps+1)):
            y0_hat_logits = model(x, updated_y, adj, torch.tensor([i]).to(x.device), self.num_timesteps)
            y0_hat_probs = torch.softmax(y0_hat_logits / temp, dim=-1)
            y0_hat = torch.multinomial(y0_hat_probs, num_samples=1) # of shape (|V|, 1)

            p = self.diff_Y.lambda2[i].repeat(masked_flag.shape)
            should_denoised = torch.logical_and(masked_flag, torch.bernoulli(p) if i > 1 else torch.ones_like(masked_flag)) # ensure all are denoised. is this necessary?
            updated_y = updated_y + should_denoised.unsqueeze(-1) * F.one_hot(y0_hat.squeeze(1), updated_y.shape[1]).float()
            masked_flag[should_denoised] = 0
        return F.one_hot(torch.argmax(updated_y, dim = -1), updated_y.shape[1]).float()

    # Manifold-constrained sampling
    def mc_estimate(self, model, x, adj, y, mask, temp = 0.02, coef = 1, corr_dist='Unif'):
        model.eval()
        with torch.no_grad():
            updated_y = torch.zeros_like(y)
            masked_flag = torch.ones_like(mask)

            #mistake = torch.zeros_like(mask)

            for i in reversed(range(1, self.num_timesteps+1)):
                y0_hat_logits = model(x, updated_y, adj, torch.tensor([i]).to(x.device), self.num_timesteps)

                #predicted_y = torch.argmax(y0_hat_logits, dim = -1)
                #gt_y = torch.argmax(y, dim = -1)
                #corrected = (predicted_y == gt_y)
                #print("=====================================> {}".format(i))
                #print(corrected.sum().item(), corrected.shape[0], float(corrected.sum().item()) / float(corrected.shape[0]))
                #print(corrected[masked_flag].sum().item(), masked_flag.sum().item(), float(corrected[masked_flag].sum().item()) / float(masked_flag.sum().item() + 1e-6))
                #print(corrected[mistake].sum().item(), mistake.sum().item(), float(corrected[mistake].sum().item()) / float(mistake.sum().item() + 1e-6))

                y0_hat_probs = torch.softmax(y0_hat_logits / temp, dim=-1)
                y0_hat = torch.multinomial(y0_hat_probs, num_samples=1) # of shape (|V|, 1)
                y0_hat = F.one_hot(y0_hat.squeeze(1), updated_y.shape[1]).float()

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
                    elif corr_dist.endswith('UM'):
                        lgst_prob, _ = torch.softmax(y0_hat_logits, dim=-1).max(-1)
                        remaining_indices = remaining_indices[torch.randperm(remaining_indices.size(0))]
                        val_arr = lgst_prob[remaining_indices]
                        top_k_vals, top_k_indices = torch.topk(val_arr, int(num_should_denoised.item()))
                        picked_indices = remaining_indices[top_k_indices] if i > 1 else remaining_indices # ensure all are denoised. is this necessary?
                    else:
                        raise ValueError(corr_dist)

                    should_denoised[picked_indices] = 1

                #print(torch.logical_and(should_denoised, mask).sum(), should_denoised.sum())

                if i > 1:
                    should_refine = torch.logical_and(should_denoised, mask)
                    y0_hat[should_refine] = y[should_refine]

                updated_y = updated_y + should_denoised.unsqueeze(-1) * y0_hat
                masked_flag[should_denoised] = 0

                #mistake = torch.logical_or(mistake, torch.logical_and(~corrected, torch.logical_and(should_denoised, ~mask)))

            return F.one_hot(torch.argmax(updated_y, dim = -1), updated_y.shape[1]).float()

    def sample_time(self, device):
        t = torch.randint(1, self.num_timesteps+1, (1,), device=device[0]).long()
        return t


class discrete_dpm_losses:
    def __init__(self, num_timesteps, y_classes, marginal, noise_schedule, device):
        # self.diff_Y = diffusion_model(device=device, timesteps = num_timesteps)
        self.num_timesteps = num_timesteps
        self.y_classes = y_classes
        self.device = device
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(noise_schedule=noise_schedule, timesteps=num_timesteps, device=device[0])
        if marginal is not None:
            self.transition_model = DiscreteMarginalTransition(y_classes=y_classes, marginal=marginal)
            self.limit_dist = marginal
        else:
            self.transition_model = DiscreteUniformTransition(y_classes=y_classes)
            self.limit_dist = torch.ones(self.y_classes) / self.y_classes

    def loss_fn(self, model, x, adj, y, train_mask, batch = 1):
        losses = None
        for i in range(0, batch):
            # t = self.sample_time(self.device)
            t = torch.randint(1, self.num_timesteps + 1, size=(1, 1), device=self.device[0]).long()
            t_float = t.float() / self.num_timesteps
            alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)
            q_y = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device[0])
            probY = y @ q_y.squeeze(0)
            Y_t = sample_discrete_features(probY=probY)

            pred_y = model(x, Y_t, adj, t.squeeze(-1), self.num_timesteps, train=True)

            # Compute losses for observed nodes/home/jhs/DPM-GSP-semi-supervised/models
            if losses == None:
                losses = CELoss(pred_y[train_mask], y[train_mask])
            else:
                losses = losses + CELoss(pred_y[train_mask], y[train_mask])

            # Compute losses for unobserved nodes
            losses = losses + CELoss(pred_y[~train_mask], y[~train_mask])

        return losses/batch

    def estimate(self, model, x, adj, y, mask, temp=0.001):
        y_t = sample_discrete_feature_noise(limit_dist=self.limit_dist, y=y)

        for s_int in reversed(range(0, self.num_timesteps)):
            s_array = s_int * torch.ones((1, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.num_timesteps
            t_norm = t_array / self.num_timesteps

            # Sample z_s
            y_t, prob_y = self.sample_p_zs_given_zt(model, s_norm, t_norm, x, adj, y_t, y, mask)

        # return F.one_hot(torch.argmax(prob_y, dim = -1), prob_y.shape[1]).float()
        return y_t


    # Manifold-constrained sampling
    def mc_estimate(self, model, x, adj, y, mask, temp = 0.0001, coef = 0.2):
        y_t = sample_discrete_feature_noise(limit_dist=self.limit_dist, y=y)
        batch_size = 1
        for s_int in reversed(range(0, self.num_timesteps)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.num_timesteps
            t_norm = t_array / self.num_timesteps

            y_t, prob_y = self.sample_p_zs_given_zt(model, s_norm, t_norm, x, adj, y_t, y, mask, coef, MCS=True)  # (n, dy)

            if s_int > 0:
                alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_norm)
                q_y = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device[0])
                probY = y[mask] @ q_y.squeeze(0)
                y_labeled = sample_discrete_features(probY=probY)
                y_t[mask] = y_labeled

            # else:
            #     y_t = F.one_hot(torch.argmax(prob_y, dim = -1), prob_y.shape[1]).float()

        return y_t


    # Manifold-constrained sampling
    def mc_monte_estimate(self, model, x, adj, y, mask, temp = 0.0001, coef = 1, test_mask = None):
        pass

    def sample_p_zs_given_zt(self, model, s, t, x, adj, y_t, y, mask, coef = 1, MCS=False):  # MCS - manifold constrained sampling
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""

        n, dy = y_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb_y = self.transition_model.get_Qt_bar(alpha_t_bar, self.device[0])
        Qsb_y = self.transition_model.get_Qt_bar(alpha_s_bar, self.device[0])
        Qt_y = self.transition_model.get_Qt(beta_t, self.device[0])
        y_t = Variable(y_t, requires_grad=True)

        with torch.set_grad_enabled(True):
            # Neural net predictions
            pred_y = model(x, y_t, adj, t.squeeze(-1), self.num_timesteps)
            imp_loss = CELoss(pred_y[mask], y[mask])
            imp_loss.backward()

        pred_y = F.softmax(pred_y, dim=-1)
        pred_y = pred_y.unsqueeze(0)            # bs, n, d0
        p_s_and_t_given_0_y = compute_batched_over0_posterior_distribution(X_t=y_t.unsqueeze(0),
                                                                           Qt=Qt_y,
                                                                           Qsb=Qsb_y,
                                                                           Qtb=Qtb_y)

        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_y = pred_y.unsqueeze(-1) * p_s_and_t_given_0_y         # bs, n, d0, d_t-1
        unnormalized_prob_y = weighted_y.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_y = unnormalized_prob_y.squeeze(0)

        unnormalized_prob_y[unnormalized_prob_y <= 0] = 1e-5
        unnormalized_prob_y[torch.sum(unnormalized_prob_y, dim=-1) <= 0] = 1e-5
        prob_y = unnormalized_prob_y / torch.sum(unnormalized_prob_y, dim=-1, keepdim=True)  # bs, n, d_t-1

        if MCS and s.max() > 0:
            denominator = torch.norm(y_t.grad.data, p=2, dim=-1, keepdim=True)
            denominator[denominator == 0] = 1e-5
            prob_y = prob_y - coef * y_t.grad.data / denominator
            prob_y[prob_y <= 0] = 1e-5
            prob_y = prob_y / torch.sum(prob_y, dim=-1, keepdim=True)

        assert ((prob_y.sum(dim=-1) - 1).abs() < 1e-4).all()
        y_s = sample_discrete_features(probY=prob_y)
        assert (y_t.shape == y_s.shape)

        return y_s.type_as(y_t), prob_y

    def sample_time(self, device):
        t = torch.randint(1, self.num_timesteps+1, (1,), device=device[0]).long()
        return t

class continuous_dpm_losses:
    def __init__(self, num_timesteps, device):
        self.diff_Y = diffusion_model(device=device, timesteps = num_timesteps)
        self.num_timesteps = num_timesteps
        self.device = device

    # Loss function
    def loss_fn(self,model, x, adj, y, train_mask, batch = 1):
        losses = None

        for i in range(0, batch):
            t = self.sample_time(self.device)
            q_Y_sample, noise = self.diff_Y.q_sample(y, t)
            pred_y = model(x, q_Y_sample, adj, t, self.num_timesteps, train=True)

            # Compute losses for observed nodes/home/jhs/DPM-GSP-semi-supervised/models
            if losses == None:
                losses = torch.mean(
                    torch.sum(((pred_y[train_mask]-noise[train_mask])**2), dim = -1)
                )
            else:
                losses = losses + torch.mean(
                    torch.sum(((pred_y[train_mask]-noise[train_mask])**2), dim = -1)
                )

            # Compute losses for unobserved nodes
            losses = losses + torch.mean(
                torch.sum(((pred_y[~train_mask]-noise[~train_mask])**2), dim = -1)
            )

        return losses/batch

    def estimate(self, model, x, adj, y, mask, temp=0.0001):
        updated_y = torch.randn_like(y)*temp
        for i in range(0, self.num_timesteps-1):
            eps = model(x, updated_y, adj, torch.tensor([self.num_timesteps-i]).to(x.device), self.num_timesteps)
            updated_y = (1/self.diff_Y.sqrt_alphas[self.num_timesteps-i])*(updated_y- (self.diff_Y.thresh[self.num_timesteps-i])*eps)
        return F.one_hot(torch.argmax(updated_y, dim = -1), updated_y.shape[1]).float()

    # Manifold-constrained sampling
    def mc_estimate(self, model, x, adj, y, mask, temp = 0.0001, coef = 1):
        updated_y = torch.randn_like(y)*temp
        for i in range(0, self.num_timesteps):
            updated_y = Variable(updated_y, requires_grad=True)

            # Compute y_prime
            eps = model(x, updated_y, adj, torch.tensor([self.num_timesteps-i]).to(x.device), self.num_timesteps)
            y_prime = (1/self.diff_Y.sqrt_alphas[self.num_timesteps-i])*(updated_y- (self.diff_Y.thresh[self.num_timesteps-i])*eps)
            y_prime = y_prime + temp*torch.sqrt(self.diff_Y.posterior_variance[self.num_timesteps-i])*torch.randn_like(y_prime)

            # Compute y_hat
            score = -(1/torch.sqrt(1-self.diff_Y.alphas_cumprod[self.num_timesteps-i]))*eps
            y_hat = (1/torch.sqrt(self.diff_Y.alphas_cumprod[self.num_timesteps-i]))*(updated_y+(1-self.diff_Y.alphas_cumprod[self.num_timesteps-i])*score)

            if self.num_timesteps-i > 1:
                # Apply manifold-constrained gradient
                imp_loss = torch.sum(torch.sum(((y-y_hat)[mask])**2, dim=1))
                imp_loss.backward()
                alpha = coef/(torch.sum(torch.sum((y-y_hat)[mask]**2, dim = 1)))
                y_update = y_prime - alpha*updated_y.grad.data   # why divide the l2

                # Apply consistency step
                y_update[mask] = self.diff_Y.alphas_cumprod[self.num_timesteps-i]*y[mask] + self.diff_Y.sqrt_one_minus_alphas_cumprod[self.num_timesteps-i] *temp* torch.randn_like(y[mask])
                updated_y = y_update
            else:
                updated_y = y_prime

        return F.one_hot(torch.argmax(updated_y, dim = -1), updated_y.shape[1]).float()


    # Manifold-constrained sampling
    def mc_monte_estimate(self, model, x, adj, y, mask, temp = 0.0001, coef = 1, test_mask = None):

        inter_res = []
        inter = [0,1,3,7,15,31,63,127,255,511,1023]
        accum, results = None, None
        for m in range(0,1024):
            model.train()
            updated_y = torch.randn_like(y)*temp
            for i in range(0, self.num_timesteps):
                updated_y = Variable(updated_y, requires_grad=True)

                # Compute y_prime
                eps = model(x, updated_y, adj, torch.tensor([self.num_timesteps-i]).to(x.device), self.num_timesteps)
                y_prime = (1/self.diff_Y.sqrt_alphas[self.num_timesteps-i])*(updated_y- (self.diff_Y.thresh[self.num_timesteps-i])*eps)
                y_prime = y_prime + temp*torch.sqrt(self.diff_Y.posterior_variance[self.num_timesteps-i])*torch.randn_like(y_prime)

                # Compute y_hat
                score = -(1/torch.sqrt(1-self.diff_Y.alphas_cumprod[self.num_timesteps-i]))*eps
                y_hat = (1/torch.sqrt(self.diff_Y.alphas_cumprod[self.num_timesteps-i]))*(updated_y+(1-self.diff_Y.alphas_cumprod[self.num_timesteps-i])*score)

                if self.num_timesteps-i > 1:
                    # Apply manifold-constrained gradient
                    imp_loss = torch.sum(torch.sum(((y-y_hat)[mask])**2, dim=1))
                    imp_loss.backward()
                    alpha = coef/(torch.sum(torch.sum((y-y_hat)[mask]**2, dim = 1)))
                    y_update = y_prime - alpha*updated_y.grad.data

                    # Apply consistency step
                    y_update[mask] = self.diff_Y.alphas_cumprod[self.num_timesteps-i]*y[mask] + self.diff_Y.sqrt_one_minus_alphas_cumprod[self.num_timesteps-i] *temp* torch.randn_like(y[mask])
                    updated_y = y_update
                else:
                    updated_y = y_prime

            aa = F.one_hot(torch.argmax(updated_y.detach(), dim = -1), updated_y.shape[1]).float()
            del updated_y

            if results == None:
                results = aa
            else:
                results = results + aa

            if m in inter:
                pred, label = torch.argmax(results, dim = -1), torch.argmax(y, dim = -1)
                test_acc = torch.mean((pred==label)[test_mask].float()).item()
                inter_res.append(test_acc)

        return inter_res


    def sample_time(self, device):
        t = torch.randint(1, self.num_timesteps+1, (1,), device=device[0]).long()
        return t


class simple_losses:
    def __init__(self, device):
        self.device = device

    def loss_fn(self, model, x, adj, y, train_mask, batch = 1):
        pred_y = model(x, adj)
        losses = F.nll_loss(pred_y[train_mask], torch.argmax(y[train_mask], dim = -1))
        return losses

    def estimate(self,model, x, adj, y, train_mask, temp = 1.0):
        return model(x, adj)
