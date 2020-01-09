# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.data import TensorDataset, DataLoader
from matplotlib.ticker import FormatStrFormatter
from scipy.stats.mstats import mquantiles
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import numpy as np
import pandas as pd
import argparse
import torch


def train_test_split(x, y, p_tr=0.5):
    """
    Splits X, Y into training and testing splits
    """
    perm = torch.randperm(x.size(0))
    n_tr = int(x.size(0) * p_tr)
    x_tr = x[perm[:n_tr]]
    y_tr = y[perm[:n_tr]]
    x_te = x[perm[n_tr:]]
    y_te = y[perm[n_tr:]]

    return x_tr, y_tr, x_te, y_te


class PinballLoss(torch.nn.Module):
    """
    Quantile regression loss
    """
    def __init__(self):
        super(PinballLoss, self).__init__()

    def forward(self, yhat, y, tau=0.5):
        diff = yhat - y
        mask = diff.ge(0).float() - tau
        return (mask * diff).mean()


class TauNet(torch.nn.Module):
    def __init__(self, nh=128):
        super(TauNet, self).__init__()
        self.net = torch.nn.Sequential(
                       torch.nn.Linear(2, nh),
                       torch.nn.ReLU(),
                       torch.nn.Linear(nh, 1))

    def forward(self, x, tau):
        if type(tau) == float or type(tau) == int or tau.dim() == 0:
            tau = torch.zeros(x.size(0), 1).fill_(tau)

        return self.net(torch.cat((x, tau), 1))


def train_net(x_all,
              y_all,
              n_epochs=5000,
              nh=128,
              bs=64,
              lr=1e-3,
              wd=1e-3):

    net = TauNet(nh)

    opt = torch.optim.Adam(net.parameters(),
                           lr=lr,
                           weight_decay=wd)

    loader = DataLoader(TensorDataset(x_all, y_all),
                        shuffle=True,
                        batch_size=bs)

    loss = PinballLoss()

    for epoch in range(n_epochs):
        for x, y in loader:
            tau = torch.zeros(x.size(0), 1).uniform_(0, 1)
            iteration_loss = loss(net(x, tau), y, tau)

            opt.zero_grad()
            iteration_loss.backward()
            opt.step()

    return net

def get_quantile(x, y, grd, n_epochs=500):
    """
    Trains TauNN and returns estimated quantiles for each tau requested in 'grd'
    """
    x = torch.Tensor(x.values).reshape(-1, 1)
    y = torch.Tensor(y.values).view(-1, 1)
    net = train_net(x, y, n_epochs=n_epochs)

    qe = np.zeros(shape=(len(grd), x.shape[0]))
    for cnt, tau in enumerate(grd):
        y_hat = net(x, tau).detach().numpy().reshape(-1)
        qe[cnt] = y_hat

    return qe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantile Treatment Effect STAR dataset')
    parser.add_argument('--repetitions', type=int, default=30)
    parser.add_argument('--n_epochs', type=int, default=5000)
    args = parser.parse_args()

    grd = torch.linspace(0.1, 0.9, 10)
    n_rep = args.repetitions
    n_epc = args.n_epochs

    np.random.seed(7)
    torch.manual_seed(7)

    kindergarden = pd.read_csv('../data/kdg.csv')

    # treatment -> small class sizes (13 - 17 children)
    treated = kindergarden[kindergarden['cltype'] == "small"]

    # control -> regular class sizes (22 - 25 children)
    untreated = kindergarden[kindergarden['cltype'] == "reg"]

    qte_grd = np.zeros(shape=(n_rep, len(grd)))
    for i in range(n_rep):
        print("Running rep #%d" % i)

        small = treated.sample(n=treated.shape[0], replace=False)
        xt = treated['exp']
        yt = treated['ach_score']
        tr_q = get_quantile(xt, yt, grd, n_epochs=n_epc)

        reg = untreated.sample(n=untreated.shape[0], replace=False)
        xt = untreated['exp']
        yt = untreated['ach_score']
        utr_q = get_quantile(xt, yt, grd, n_epochs=n_epc)

        for cnt, tau in enumerate(grd):
            qte_grd[i, cnt] = np.mean(tr_q[cnt]) - np.mean(utr_q[cnt])

    mean, std = np.mean(qte_grd, axis=0), np.std(qte_grd, axis=0)

    print('QTE per requested quantile: mean(std)')
    for tau, m, s in zip(grd, mean, std):
        print('$\\tau = ' + str(tau) + ': ' + str(round(m, 2)) +
              '(' + str(round(s, 2)) + ')' + '$')

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18
    plt.figure(figsize=(12, 7))

    ax = sns.boxplot(data=qte_grd, palette="Blues")
    ax.set_xticklabels( np.around(np.linspace(0.1, 0.9, 10),  decimals=1))
    plt.xlabel('$\\tau$')
    plt.ylabel('Quantile Treatment Effect ')

    plt.show()
    ax.figure.savefig('../figures/heterogeneous_qte_' + str(n_rep) + '_' + str(n_epc) + '.pdf')
