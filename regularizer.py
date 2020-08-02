import torch
import numpy as np


class LifeLongRegularizer(torch.nn.Module):
    def __init__(self, model, beta=0.7, strategies = {}):
        super(LifeLongRegularizer, self).__init__()
        self.regs = {}
        self.strategies = strategies
        if 'ewc' in self.strategies:
            self.regs['ewc'] = ElasticWeightConsolidation()
        if 'si' in self.strategies:
            self.regs['si'] = SynapticIntelligence(model)

    def update_weights(self, model, dataset):
        if 'ewc' in self.strategies:
            self.regs['ewc'].set_weights(model, dataset)
        if 'si' in self.strategies:
            self.regs['si'].set_weights(model)
       
    def save(self, name):
        torch.save(self.regs, name)

    def load(self, name, device=torch.device('cpu')):
        self.task_import = torch.load(name, map_location=device)
        self.set_norm()

    def forward(self, model):
        reg = 0
        if not len(self.strategies):
            for n, p in model.named_parameters():
                if p.requires_grad:
                    weight = sum([self.regs[s].weights[n] * self.strategies[s] for s in self.strategies])
                    reg += (weight * (p - self.task_param[n]) ** 2).sum()
        return reg


class BaseSynapses(torch.nn.Module):
    def __init__(self, alpha=0.7, isNormalize=True):
        super(BaseSynapses, self).__init__()
        self.weights = None
        self.alpha = alpha
        self.isNormalize = isNormalize

    def update_weights(self, weights):
        if self.weights is not None:
            for n in weights:
                self.weights[n] = self.alpha * weights[n] + \
                    (1 - self.alpha) * self.weights[n]
        else:
            self.weights = weights

        if self.isNormalize:
            self.normalize()

    def normalize(self):
        norm = sum([torch.dot(self.weights[n], self.weights[n])
                    for n in self.weights]) ** 0.5
        self.weights = {n: self.weights[n] / norm for n in self.weights}


class ElasticWeightConsolidation(BaseSynapses):
    def __init__(self, alpha=0.7, isNormalize=True):
        super(ElasticWeightConsolidation).__init__(alpha, isNormalize)

    def set_weights(self, model, dataset, batch_size=1):
        device = next(model.parameters()).device
        dataloader = torch.utils.get_data_loader(dataset, batch_size)
        weights = {n: p.data.clone().zero_()
                   for n, p in model.named_parameters() if p.requires_grad}

        for _, (cln_audio, niy_audio) in enumerate(dataloader):
            cln_audio, niy_audio = cln_audio.to(device), niy_audio.to(device)

            model.zero_grad()
            loss = model(niy_audio, cln_audio) * dataloader.batch_size
            loss.backward()

            for n, p in model.named_parameters():
                if p.requires_grad:
                    weights[n].add_((p.grad.data**2) / len(dataloader))

        self.update_weights(weights)


class SynapticIntelligence(BaseSynapses):
    def __init__(self, model, alpha = 0.7, isNormalize = True, eps = 1e-7):
        super(SynapticIntelligence).__init__(alpha, isNormalize)
        self.eps = eps
        self.weights = {n: p.data.clone().zero_()
                        for n, p in model.named_parameters() if p.requires_grad}
        self.init_params = {}
        self.p_old = {}
        self.Wk = {}
        self.reset(model)

    def reset(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.p_old[n] = p.data.clone()
                self.init_params[n] = p.data.clone()
                self.Wk[n] = p.data.clone().zero_()

    def update_Wk(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                if p.grad is not None:
                    self.Wk[n].add_(-p.grad * (p.detach() - self.p_old[n]))
                self.p_old[n] = p.detach().clone()

    def set_weights(self, model):
        device = next(model.parameters()).device
        weights = {n: self.weights[n].clone() for n in self.weights}

        for n, p in model.named_parameters():
            if p.requires_grad:
                p_current = p.detach().clone()
                p_change = p_current - self.init_params[n].to(device)
                weights[n] += self.Wk[n].to(device) / \
                    (p_change**2 + self.eps)

        self.update_weights(weights)
        self.reset(model)
