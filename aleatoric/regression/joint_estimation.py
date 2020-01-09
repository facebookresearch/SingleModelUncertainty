# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

%matplotlib inline
from matplotlib import pyplot as plt
import torch
import math

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
    elif type(tau) == float:
        tau = torch.zeros(x.size(0), 1).fill_(tau)
        
    return torch.cat((x, (tau - 0.5) * 12), 1)

def train_net(x, y, q="all"):
    net = torch.nn.Sequential(
            torch.nn.Linear(d + 1, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1))

    opt = torch.optim.Adam(net.parameters(),
                           1e-3,
                           weight_decay=1e-2)
    loss = QuantileLoss()

    for _ in range(10000):
        opt.zero_grad()
        if q == "all":
            taus = torch.rand(x.size(0), 1)
        else:
            taus = torch.zeros(x.size(0), 1).fill_(q)
        loss(net(augment(x, taus)), y, taus).backward()
        opt.step()
        
    return net


n = 1000
d = 5

x = torch.randn(n, d)
y = x[:, 0].view(-1, 1).mul(5).cos() + 0.3 * torch.randn(n, 1)

net = train_net(x, y, "all")
net_01 = train_net(x, y, 0.1)
net_05 = train_net(x, y, 0.5)
net_09 = train_net(x, y, 0.9)

x = torch.randn(n, d)
y = x[:, 0].view(-1, 1).mul(5).cos() + 0.3 * torch.randn(n, 1)
o = torch.sort(x[:, 0])[1]

plt.rc('text', usetex=True)
plt.rc('font', size=16)
plt.rc('text.latex', preamble=r'\usepackage{times}')
plt.figure(figsize=(15, 4))
plt.rc('font', family='serif')

plt.subplot(1, 4, 1)
plt.title("separate estimation")
plt.plot(x[o, 0].numpy(), y[o].detach().numpy(), '.')
plt.plot(x[o, 0].numpy(), net_01(augment(x[o], 0.1)).detach().numpy(), alpha=0.75, label="$\\tau_{0.1}$")
plt.plot(x[o, 0].numpy(), net_05(augment(x[o], 0.5)).detach().numpy(), alpha=0.75, label="$\\tau_{0.5}$")
plt.plot(x[o, 0].numpy(), net_09(augment(x[o], 0.9)).detach().numpy(), alpha=0.75, label="$\\tau_{0.9}$")
plt.legend()

plt.subplot(1, 4, 2)

plt.plot(x[o, 0].numpy(), (net_09(augment(x[o], 0.9)) - net_05(augment(x[o], 0.5))).detach().numpy(), alpha=0.75, label="$\\tau_{0.9} - \\tau_{0.5}$")
plt.plot(x[o, 0].numpy(), (net_05(augment(x[o], 0.5)) - net_01(augment(x[o], 0.1))).detach().numpy(), alpha=0.75, label="$\\tau_{0.5} - \\tau_{0.1}$")
plt.axhline(0, ls="--", color="gray")
plt.legend()

plt.subplot(1, 4, 3)
plt.title("joint estimation")
plt.plot(x[o, 0].numpy(), y[o].detach().numpy(), '.')
plt.plot(x[o, 0].numpy(), net(augment(x[o], 0.1)).detach().numpy(), alpha=0.75, label="$\\tau_{0.1}$")
plt.plot(x[o, 0].numpy(), net(augment(x[o], 0.5)).detach().numpy(), alpha=0.75, label="$\\tau_{0.5}$")
plt.plot(x[o, 0].numpy(), net(augment(x[o], 0.9)).detach().numpy(), alpha=0.75, label="$\\tau_{0.9}$")
plt.legend()

plt.subplot(1, 4, 4)
plt.plot(x[o, 0].numpy(), (net(augment(x[o], 0.9)) - net(augment(x[o], 0.5))).detach().numpy(), alpha=0.75, label="$\\tau_{0.9} - \\tau_{0.5}$")
plt.plot(x[o, 0].numpy(), (net(augment(x[o], 0.5)) - net(augment(x[o], 0.1))).detach().numpy(), alpha=0.75, label="$\\tau_{0.9} - \\tau_{0.5}$")

plt.axhline(0, ls="--", color="gray")
plt.legend()

plt.tight_layout(0, 0, 0)
plt.savefig("joint_estimation.pdf")
