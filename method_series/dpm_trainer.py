import os
import time
from tqdm import tqdm, trange
import numpy as np
import torch
import random
import torch.nn.functional as F
from utils.loader import load_seed, load_device, load_data, load_model_params, load_model_optimizer, load_loss_fn, \
                         load_simple_model_optimizer, load_simple_loss_fn
from utils.logger import Logger, set_log, start_log, train_log

import scipy.sparse as sp
import copy
from torch_scatter import scatter
from torch_geometric.utils import is_undirected, to_undirected, add_self_loops, degree

class Trainer(object):
    def __init__(self, config):
        super(Trainer, self).__init__()

        self.config = config
        self.log_folder_name, self.log_dir = set_log(self.config)
        self.seed = load_seed(self.config.seed)
        self.device = load_device()
        self.x, self.y, self.adj, self.train_mask, self.valid_mask, self.test_mask = load_data(self.config) # self.y is of shape (|V|, #Class)
        self.losses = load_loss_fn(self.config, self.device, self.y[self.train_mask])
        self.simple_losses = load_simple_loss_fn(self.config, self.device)

    def train(self, ts):
        self.config.exp_name = ts
        self.ckpt = f'{ts}'
        print('\033[91m' + f'{self.ckpt}' + '\033[0m')

        # Prepare model, optimizer, and logger
        self.params = load_model_params(self.config)
        self.simple_model, self.simple_optimizer, self.simple_scheduler = load_simple_model_optimizer(self.params, self.config.train, self.device)
        self.model, self.optimizer, self.scheduler = load_model_optimizer(self.params, self.config.train, self.device)
        self.loss_fn = self.losses.loss_fn
        self.simple_loss_fn = self.simple_losses.loss_fn
        self.estimator = self.losses.estimate
        self.mc_estimator = self.losses.mc_estimate
        #self.mc_monte_estimate = self.losses.mc_monte_estimate
        self.simple_estimator = self.simple_losses.estimate
        total_params = sum(p.numel() for p in self.model.parameters())
        print(total_params)
        logger = Logger(str(os.path.join(self.log_dir, f'{self.ckpt}.log')), mode='a')
        logger.log(f'{self.ckpt}', verbose=False)
        start_log(logger, self.config)
        train_log(logger, self.config)

        root = os.path.join("./pretrain-GNN", self.config.model.model)
        # parent = os.path.join(root, f"{self.config.data.data}-{self.config.model.num_layers} layers-{self.config.model.nhid} dim")
        parent = os.path.join(root, f"{self.config.data.data}")
        if not os.path.exists(parent):
            os.makedirs(parent)

        edge_index = to_undirected(self.adj)
        edge_index = add_self_loops(edge_index)[0]
        ckpt_path = os.path.join(parent, f'seed_{self.config.seed}.ckpt')

        best_valid, test_at_that_point, best_est, base_hetero = 0, 0, None, 0
        ckpt = None
        # if self.config.data.data != 'Amazon-ratings' and os.path.exists(ckpt_path):
        if os.path.exists(ckpt_path):
            print('load pretrained mean-field GNN...')
            ckpt = torch.load(ckpt_path)
            self.simple_model.load_state_dict(ckpt)

            with torch.no_grad():
                self.simple_model.eval()
                y_est = self.simple_estimator(self.simple_model, self.x, self.adj, self.y, self.train_mask)
                pred = torch.argmax(y_est, dim = -1)
                label = torch.argmax(self.y, dim = -1)
                valid_acc = torch.mean((pred==label)[self.valid_mask].float()).item()
                test_acc = torch.mean((pred==label)[self.test_mask].float()).item()

                if valid_acc > best_valid:
                    best_valid = valid_acc
                    test_at_that_point = test_acc
                    best_est = y_est / self.config.diffusion.simple_temp


                # subgraph accuracy
                res = (pred==label).long()
                assert res.size(0) == self.x.size(0)

                src = res[edge_index[0]]
                num_valid = scatter(src, edge_index[1], reduce='sum')
                num_neighbors = degree(edge_index[0], dtype=torch.long)
                val_sub_acc = torch.mean((num_valid == num_neighbors)[self.valid_mask].float()).item()
                test_sub_acc = torch.mean((num_valid == num_neighbors)[self.test_mask].float()).item()
                print(f'pretrain_val_sub_acc={val_sub_acc :.6f} \t pretrain_test_sub_acc={test_sub_acc :.6f}\n')
                # with open(f'/root/graph_research/DPM-SNC-Discrete/transductive-node-classification/results_hpo/{self.config.data.data}_pretrain_sub_acc.out', 'a') as f:
                #     f.write(f'val_sub_acc={val_sub_acc :.6f} \t test_sub_acc={test_sub_acc :.6f}\n')

                # exit()


        else:
            # Pre-train mean-field GNN
            print('Pretrain mean-field GNN...')
            for i in range(0,self.config.train.pre_train_epochs):
                self.simple_model.train()
                self.simple_optimizer.zero_grad()

                loss_subject = (self.x, self.adj, self.y, self.train_mask)
                loss = self.simple_loss_fn(self.simple_model, *loss_subject)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.simple_model.parameters(), self.config.train.grad_norm)
                self.simple_optimizer.step()
                if self.config.train.lr_schedule:
                    self.simple_scheduler.step()


                # Evaluate mean-field GNN
                if i%10 == 0:
                    with torch.no_grad():
                        self.simple_model.eval()
                        y_est = self.simple_estimator(self.simple_model, self.x, self.adj, self.y, self.train_mask)
                        pred = torch.argmax(y_est, dim = -1)
                        label = torch.argmax(self.y, dim = -1)
                        valid_acc = torch.mean((pred==label)[self.valid_mask].float()).item()
                        test_acc = torch.mean((pred==label)[self.test_mask].float()).item()

                        if valid_acc > best_valid:
                            best_valid = valid_acc
                            test_at_that_point = test_acc
                            best_est = y_est / self.config.diffusion.simple_temp
                            ckpt = copy.deepcopy(self.simple_model.state_dict())

            torch.save(ckpt, ckpt_path)

        print('Done!')
        print(f'Pre-train | best val: {best_valid:.3f} | best test: {test_at_that_point:.3f}', end = '\n')


        # Prepare expectation step
        buffer, n_samples, buffer_size, priority = None, 3, 50, None
        xs, adjs, ys, best_ests, masks = [], [], [], [], []
        for i in range(0,n_samples):
            xs.append(self.x)
            adjs.append(self.adj+self.x.shape[0]*i)
            ys.append(self.y)
            best_ests.append(best_est)
            masks.append(self.train_mask)
        xs, adjs, ys, masks = torch.cat(xs, dim = 0), torch.cat(adjs, dim = 1), torch.cat(ys, dim = 0), torch.cat(masks, dim = 0) # (n_samples*number of data, )
        best_prob = torch.exp(torch.cat(best_ests, dim = 0)) # best_est corresponds to the results of F.log_softmax()


        # Train the model
        best_valid, best_test, best_sub_val, best_sub_test = 0, 0, 0, 0
        for epoch in range(0, self.config.train.num_epochs):
            t_start = time.time()

            # Expectation step
            if epoch % self.config.train.load_interval == 0:

                if epoch > self.config.train.load_start: # Use manifold-constarined sampling of DPM-GSP
                    if self.config.diffusion.method == 'LP':
                        expected_y_set = self.mc_estimator(self.model, xs, adjs, ys, masks, temp = self.config.diffusion.temp, coef = self.config.diffusion.coef, corr_dist=self.config.diffusion.estep_corr_dist) # TODO: is it necessary to make this consistent with evaluation?
                    else:
                        expected_y_set = self.mc_estimator(self.model, xs, adjs, ys, masks, temp = self.config.diffusion.temp, coef = self.config.diffusion.coef)
                else: # Use mean-field GNN
                    expected_y_set = torch.distributions.categorical.Categorical(best_prob).sample()
                    expected_y_set = F.one_hot(expected_y_set, best_prob.shape[1]).float()

                # expected_y_set = expected_y_set.cpu()
                label = torch.argmax(self.y, dim = -1)
                cur_estep_val_accs, cur_estep_test_accs = [], []
                cur_estep_val_subaccs, cur_estep_test_subaccs = [], []
                for i in range(0, n_samples):
                    pred = torch.argmax(expected_y_set[i*self.y.shape[0]:(i+1)*self.y.shape[0]], dim=-1)
                    valid_acc = torch.mean((pred==label)[self.valid_mask].float()).item()
                    test_acc = torch.mean((pred==label)[self.test_mask].float()).item()
                    cur_estep_val_accs.append(valid_acc)
                    cur_estep_test_accs.append(test_acc)

                    # subgraph accuracy
                    res = (pred==label).long()
                    assert res.size(0) == self.x.size(0)

                    src = res[edge_index[0]]
                    num_valid = scatter(src, edge_index[1], reduce='sum')
                    num_neighbors = degree(edge_index[0], dtype=torch.long)
                    val_sub_acc = torch.mean((num_valid == num_neighbors)[self.valid_mask].float()).item()
                    test_sub_acc = torch.mean((num_valid == num_neighbors)[self.test_mask].float()).item()
                    cur_estep_val_subaccs.append(val_sub_acc)
                    cur_estep_test_subaccs.append(test_sub_acc)
                #print(cur_estep_val_accs, cur_estep_test_accs)

                print("{} Estep: val = {} ({}), test = {} ({})".format(epoch, np.mean(cur_estep_val_accs), np.std(cur_estep_val_accs), np.mean(cur_estep_test_accs), np.std(cur_estep_test_accs)))
                print("{} Estep: val_sub = {} ({}), test_sub = {} ({})".format(epoch, np.mean(cur_estep_val_subaccs), np.std(cur_estep_val_subaccs), np.mean(cur_estep_test_subaccs), np.std(cur_estep_test_subaccs)))

                # Fill the buffer
                expected_y_set = torch.cat([expected_y_set[i*self.y.shape[0]:(i+1)*self.y.shape[0]].view(1,self.y.shape[0],-1) for i in range(0,n_samples)], dim = 0) # (n_samples, number of data, number of classes)

                if buffer == None:
                    buffer = expected_y_set
                    if self.config.train.priority_queue:
                        priority = cur_estep_val_accs
                        #priority = cur_estep_val_subaccs

                else:
                    buffer = torch.cat([buffer,expected_y_set], dim = 0)
                    if self.config.train.priority_queue:
                        priority.extend(cur_estep_val_accs)
                        #priority.extend(cur_estep_val_subaccs)

                start = buffer.shape[0]-buffer_size
                if start < 0:
                    start = 0
                buffer = buffer[start:]
                if self.config.train.priority_queue:
                    priority = priority[start:]

                if self.config.train.priority_queue:
                    # val_acc^\alpha * epoch * ELBO
                    prob = F.softmax(torch.tensor(priority) / self.config.diffusion.priority_temp, dim=-1)
                    # print(prob)
                    index = torch.distributions.categorical.Categorical(prob).sample()
                    y_train = buffer[index]
                else:
                    # Maximization step
                    y_train = buffer[np.random.randint(buffer.shape[0]+1)-1] # Sample from the buffer

            # print(buffer.shape)
            y_train[self.train_mask] = self.y[self.train_mask]
            # y_train = y_train.to(self.device[0])
            torch.cuda.empty_cache()

            self.model.train()
            self.optimizer.zero_grad()
            if self.config.diffusion.method == 'LP':
                loss_subject = (self.x, self.adj, y_train, self.train_mask, self.config.train.time_batch, self.config.diffusion.weightts, self.config.diffusion.emphasize_labeled, self.config.diffusion.learn_identity)
            else:
                loss_subject = (self.x, self.adj, y_train, self.train_mask, self.config.train.time_batch)
            loss = self.loss_fn(self.model, *loss_subject)
            loss.backward()
            self.optimizer.step()

            if self.config.train.lr_schedule:
                self.scheduler.step()


            # Evaluate the model
            if epoch % self.config.train.print_interval == 0 and epoch > 0:

                # temp_mc_estimate = 0.0001   # 0.02 for homo graph

                # Manifold-constrained sampling
                if self.config.diffusion.multichain: # not theoretically sound due to that each node's label is not independent in our diffusion model
                    # for comparison
                    if self.config.diffusion.method == 'LP':
                        y_est = self.mc_estimator(self.model, self.x, self.adj, self.y, self.train_mask, temp = 0.02, coef = self.config.diffusion.coef, corr_dist=self.config.diffusion.eval_corr_dist)
                    else:
                        y_est = self.mc_estimator(self.model, self.x, self.adj, self.y, self.train_mask, temp = 0.0003, coef = self.config.diffusion.coef)
                    pred, label = torch.argmax(y_est, dim = -1), torch.argmax(self.y, dim = -1)
                    valid_acc = torch.mean((pred==label)[self.valid_mask].float()).item()
                    test_acc = torch.mean((pred==label)[self.test_mask].float()).item()
                    print("Greedy single traj: {}, {}".format(valid_acc, test_acc))

                    y_ests = []
                    for mc_i in range(100):
                        if self.config.diffusion.method == 'LP':
                            y_est = self.mc_estimator(self.model, self.x, self.adj, self.y, self.train_mask, temp = self.config.diffusion.temp, coef = self.config.diffusion.coef, corr_dist=self.config.diffusion.eval_corr_dist)
                        else:
                            y_est = self.mc_estimator(self.model, self.x, self.adj, self.y, self.train_mask, temp = 0.0003, coef = self.config.diffusion.coef)

                        pred, label = torch.argmax(y_est, dim = -1), torch.argmax(self.y, dim = -1)
                        valid_acc = torch.mean((pred==label)[self.valid_mask].float()).item()
                        test_acc = torch.mean((pred==label)[self.test_mask].float()).item()
                        print("MC traj {}: {}, {}".format(mc_i, valid_acc, test_acc))
                        y_ests.append(y_est)
                    y_est = torch.cat([ts.unsqueeze(0) for ts in y_ests], axis=0).sum(axis=0)
                else:
                    if self.config.diffusion.method == 'LP':
                        y_est = self.mc_estimator(self.model, self.x, self.adj, self.y, self.train_mask, temp = 0.02, coef = self.config.diffusion.coef, corr_dist=self.config.diffusion.eval_corr_dist)
                    else:
                        y_est = self.mc_estimator(self.model, self.x, self.adj, self.y, self.train_mask, temp = 0.0003, coef = self.config.diffusion.coef)
                    pred, label = torch.argmax(y_est, dim = -1), torch.argmax(self.y, dim = -1)
                    valid_acc = torch.mean((pred==label)[self.valid_mask].float()).item()
                    test_acc = torch.mean((pred==label)[self.test_mask].float()).item()

                    # subgraph accuracy
                    res = (pred==label).long()
                    assert res.size(0) == self.x.size(0)

                    src = res[edge_index[0]]
                    num_valid = scatter(src, edge_index[1], reduce='sum')
                    num_neighbors = degree(edge_index[0], dtype=torch.long)
                    val_sub_acc = torch.mean((num_valid == num_neighbors)[self.valid_mask].float()).item()
                    test_sub_acc = torch.mean((num_valid == num_neighbors)[self.test_mask].float()).item()

                    if val_sub_acc >= best_sub_val:
                        best_sub_val, best_sub_test = val_sub_acc, test_sub_acc

                    if valid_acc >= best_valid:
                        best_valid, best_test = valid_acc, test_acc
                        #torch.save(self.model.state_dict(), 'saved_model/'+self.config.data.data+'_'+self.config.model.model+'_'+str(self.config.model.nhid)+'_'+str(self.config.model.num_layers)+'_'+str(self.config.train.lr)+'_'+str(self.config.train.weight_decay)+'_'+str(self.config.diffusion.temp)+'_'+str(self.config.seed)+'.pt')
                    with torch.no_grad():
                        y_est = self.estimator(self.model, self.x, self.adj, self.y, self.train_mask)
                    pred = torch.argmax(y_est, dim = -1)
                    train_acc = torch.mean((pred==label)[self.train_mask].float()).item()

                    # Log intermediate performance
                    logger.log(f'{epoch+1:03d} | val: {valid_acc:.3f} | test: {test_acc:.3f}  | best val: {best_valid:.3f} | best test: {best_test:.3f} | best_sub_val: {best_sub_val:.3f} | best_sub_test: {best_sub_test:.3f}', verbose=False)
                    print(f'[Epoch {epoch+1:05d}] | train: {train_acc:.3f} | val: {valid_acc:.3f} | test: {test_acc:.3f}  | best val: {best_valid:.3f} | best test: {best_test:.3f}', end = '\n')
                    print(f'[Epoch {epoch+1:05d}] | val_sub: {val_sub_acc:.3f} | test_sub: {test_sub_acc:.3f}  | best_sub_val: {best_sub_val:.3f} | best_sub_test: {best_sub_test:.3f}', end = '\n')

        if self.config.diffusion.weightts:
            obj = 0
        else:
            obj = 1

        filename = f'{self.config.model.num_layers}_layers-nhid_{self.config.model.nhid}_{self.config.diffusion.method}_{obj}_{self.config.train.load_start}_{self.config.train.lr}_{self.config.train.weight_decay}_{self.config.train.lr_diffusion}_{self.config.train.weight_decay_diffusion}_{self.config.train.load_interval}_{self.config.diffusion.temp}_{self.config.diffusion.simple_temp}_{self.config.diffusion.priority_temp}_{self.config.diffusion.estep_corr_dist}_{self.config.diffusion.eval_corr_dist}_{self.config.diffusion.step}_{self.config.train.priority_queue}'
        with open(f'/root/graph_research/DPM-SNC-Discrete/transductive-node-classification/results_ablation/{self.config.data.data}/{filename}.out', 'a') as f:
            f.write(f"seed={self.config.seed} \t acc_MFGNN={test_at_that_point} \t best_val={best_valid:.5f} \t best_test={best_test:.5f} \t best_sub_val={best_sub_val:.5f} \t best_sub_test={best_sub_test:.5f}\n")
        print(self.config)
        # print(f"mc_estimate={temp_mc_estimate}")
