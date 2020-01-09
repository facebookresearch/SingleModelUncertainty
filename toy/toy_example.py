# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader, TensorDataset

class QuantileLoss(torch.nn.Module):
    def __init__(self):
        super(QuantileLoss, self).__init__()

    def forward(self, yhat, y, tau):
        diff = yhat - y
        mask = (diff.ge(0).float() - tau).detach()
        return (mask * diff).mean()


def augment(x, tau=None):
    if tau is None:
        tau = torch.zeros(x.size(0), 1).fill_(0.5)
        
    return torch.cat((x, (tau - 0.5) * 12), 1)
    

def build_certificates(x, k=100, epochs=500):
    c = torch.nn.Linear(x.size(1), k)
        
    loader = DataLoader(TensorDataset(x),
                        shuffle=True,
                        batch_size=128)
        
    opt = torch.optim.Adam(c.parameters())

    for epoch in range(epochs):
        for xi in loader:
            opt.zero_grad()
            error = c(xi[0]).pow(2).mean()
            penalty = (c.weight @ c.weight.t() - torch.eye(k)).pow(2).mean()
            (error + penalty).backward()
            opt.step()

    return c


def simple_network(n_inputs=1, n_outputs=1, n_hiddens=100):
    return torch.nn.Sequential(
           torch.nn.Linear(n_inputs, n_hiddens),
           torch.nn.ReLU(),
           torch.nn.Linear(n_hiddens, n_hiddens),
           torch.nn.ReLU(),
           torch.nn.Linear(n_hiddens, 1))


def generate_data(n=1024):
    sep = 1
    x = torch.zeros(n // 2, 1).uniform_(0, 0.5)
    x = torch.cat((x, torch.zeros(n // 2, 1).uniform_(0.5 + sep, 1 + sep)), 0)
    m = torch.distributions.Exponential(torch.tensor([3.0]))
    noise = m.rsample((n,))
    y = (2 * 3.1416 * x).sin() + noise
    x_test = torch.linspace(-0.5, 2.5, 100).view(-1, 1)
    return x, y, x_test
    

def train_network(x, y, epochs=500):
    net = simple_network(x.size(1) + 1, y.size(1))    
    optimizer = torch.optim.Adam(net.parameters())
    
    loss = QuantileLoss()
    loader = DataLoader(TensorDataset(x, y), shuffle=True, batch_size=128)
    
    for _ in range(epochs):
        for xi, yi in loader:
            optimizer.zero_grad()
            taus = torch.rand(xi.size(0), 1)
            loss(net(augment(xi, taus)), yi, taus).backward()
            optimizer.step()
        
    return net
    
# train main network ########################################################

torch.manual_seed(0)

x, y, test_x = generate_data(1000)

test_x = (test_x - x.mean(0)) / x.std(0)
x = (x - x.mean(0)) / x.std(0)

net = train_network(x, y)
taus = torch.zeros(test_x.size(0), 1)

pred_low = net(augment(test_x, taus + 0.025)).detach().numpy().ravel()
pred_med = net(augment(test_x, taus + 0.500)).detach().numpy().ravel()
pred_hig = net(augment(test_x, taus + 0.975)).detach().numpy().ravel()

f = net[:-2](augment(x)).detach()
test_f = net[:-2](augment(test_x)).detach()
test_y = net(augment(test_x)).detach().numpy().ravel()

cert = build_certificates(f)
scores = cert(test_f).pow(2).mean(1).detach().numpy()

scores = (scores - scores.min()) / (scores.max() - scores.min()) * 3

plt.figure(figsize=(5, 3))
plt.plot(x.numpy(), y.numpy(), '.', alpha=0.15)

plt.plot(test_x.view(-1).numpy(),
         pred_med,
         color="gray",
         alpha=0.5,
         lw=2)

plt.fill_between(test_x.view(-1).numpy(),
                 pred_low,
                 pred_hig,
                 color="gray",
                 alpha=0.25,
                 label='aleatoric')

plt.fill_between(test_x.view(-1).numpy(),
                 pred_med - scores,
                 pred_med + scores,
                 color="pink",
                 alpha=0.25,
                 label='epistemic')

plt.ylim(-2, 2.75)
plt.legend(loc=3)
plt.tight_layout(0, 0, 0)
plt.savefig("toy_example.pdf")
plt.show()
