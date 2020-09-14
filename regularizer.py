import torch
import numpy as np


class LifeLongAgent(torch.nn.Module):
    def __init__(self, model, strategies={}):
        super(LifeLongAgent, self).__init__()
        assert len(strategies)
        self.regs = {}
        self.strategies = strategies
        if 'ewc' in self.strategies:
            self.regs['ewc'] = ElasticWeightConsolidation()
        if 'si' in self.strategies:
            self.regs['si'] = SynapticIntelligence(model)

    def update_weights(self, model, dataloader):
        if 'ewc' in self.strategies:
            self.regs['ewc'].set_weights(model, dataloader)
        if 'si' in self.strategies:
            self.regs['si'].set_weights(model)
        self.task_params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        
    def forward(self, model):
        reg = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                weight = sum([self.regs[s].weights[n] *
                                self.strategies[s] for s in self.strategies])
                reg += (weight * (p - self.task_params[n]).pow(2)).sum()
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
        norm = sum([self.weights[n].pow(2).sum() for n in self.weights]).sqrt()
        self.weights = {n: self.weights[n] / norm for n in self.weights}


class ElasticWeightConsolidation(BaseSynapses):
    def __init__(self, alpha=0.7, isNormalize=True):
        super(ElasticWeightConsolidation, self).__init__(alpha, isNormalize)

    def set_weights(self, model, dataloader):
        device = next(model.parameters()).device
        weights = {n: p.data.clone().zero_()
                   for n, p in model.named_parameters() if p.requires_grad}
        count = 0

        for (lengths, niy_audio, cln_audio) in dataloader:
            lengths, niy_audio, cln_audio = lengths.to(
                device), niy_audio.to(device), cln_audio.to(device)

            batch_size = len(niy_audio)
            count += batch_size

            model.zero_grad()
            loss = model(lengths, niy_audio, cln_audio) * batch_size
            loss.backward()

            for n, p in model.named_parameters():
                if p.requires_grad:
                    weights[n].add_((p.grad.data.pow(2)))

        weights = {n: weights[n] / count for n in weights}
        self.update_weights(weights)


class SynapticIntelligence(BaseSynapses):
    def __init__(self, model, alpha=0.7, isNormalize=True, eps=1e-7):
        super(SynapticIntelligence, self).__init__(alpha, isNormalize)
        self.eps = eps
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
        if self.weights is None:
            weights = {n: p.data.clone().zero_()
                        for n, p in model.named_parameters() if p.requires_grad}
        else:
            weights = {n: self.weights[n].clone() for n in self.weights}

        for n, p in model.named_parameters():
            if p.requires_grad:
                p_current = p.detach().clone()
                p_change = p_current - self.init_params[n].to(device)
                weights[n] += self.Wk[n].to(device) / \
                    (p_change.pow(2) + self.eps)

        self.update_weights(weights)
        self.reset(model)
