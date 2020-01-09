# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.stats.mstats import mquantiles
from tqdm import tqdm
import argparse
import torch
import os


def TuebingenDataset(root):
    """
    Reads the Tuebingen cause-effect dataset
    https://webdav.tuebingen.mpg.de/cause-effect/
    """
    import numpy as np

    meta = np.genfromtxt(os.path.join(root, 'pairmeta.txt'), dtype=np.str)
    samples = []
    weights = []
    names = []

    for i in range(meta.shape[0]):
        fname = 'pair' + meta[i][0] + '.txt'
        data = np.genfromtxt(os.path.join(root, fname))
        x = data[:, 0]
        y = data[:, 1]

        # remove binary pairs
        if (len(np.unique(x)) > 2) and (len(np.unique(y)) > 2):

            if((meta[i][1] == '1') and
               (meta[i][2] == '1') and
               (meta[i][3] == '2') and
               (meta[i][4] == '2')):
                d = torch.Tensor(np.vstack((x, y)).T)
                w = float(meta[i][5])
                samples.append(d)
                weights.append(w)
                names.append(fname)

            if((meta[i][1] == '2') and
               (meta[i][2] == '2') and
               (meta[i][3] == '1') and
               (meta[i][4] == '1')):
                d = torch.Tensor(np.vstack((y, x)).T)
                w = float(meta[i][5])
                samples.append(d)
                weights.append(w)
                names.append(fname)

    return samples, weights, names


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


def qnorm_scale(x):
    """
    transform variable to standard Gaussian
    """
    from scipy import stats
    import pandas

    # rank elements with method = "random" when resolving ties
    x = pandas.Series(x.numpy())
    x_rank = x.sample(frac=1).rank(method='first').reindex_like(x)/(x.size + 1)

    x = stats.norm.ppf(x_rank.values)

    return torch.Tensor(x).view(-1, 1)


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


def causal_score(net, x, y):
    """
    S(x \to y) = CL(P(X)) + CL(P(Y|X)) ~ QS(X) + QS(Y|X)

    https://arxiv.org/abs/1801.10579 [Theorem 3]
    """
    loss = PinballLoss()
    scores = []
    for tau, tau_w in zip(TAUS, wTAUS):
        x_marg = torch.Tensor(mquantiles(x.numpy(), prob=tau, alphap=1, betap=1))
        x_marg_qs = loss(x_marg, x, tau).item()  # QS(X)
        y_cond_qs = loss(net(x, tau), y, tau).item()  # QS(Y|X)

        scores.append((x_marg_qs + y_cond_qs) * tau_w)

    return scores


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


def train_net(x_tr,
              y_tr,
              x_te,
              y_te,
              n_epochs=10000,
              nh=64,
              lr=1e-4,
              wd=1e-4):

    net = TauNet(nh)

    opt = torch.optim.Adam(net.parameters(),
                           lr=lr,
                           weight_decay=wd)

    scheduler = ReduceLROnPlateau(opt,
                                  patience=2,
                                  factor=0.99,
                                  threshold=1e-10,
                                  min_lr=1e-10,
                                  threshold_mode='abs')

    loss = PinballLoss()

    for epoch in tqdm(range(n_epochs)):
        taus = torch.randn(x_tr.size(0), 1).uniform_()
        opt.zero_grad()
        loss(net(x_tr, taus), y_tr, taus).backward()
        opt.step()

        test_loss = 0
        for tau in TAUS:
            test_loss += loss(net(x_te, tau), y_te, tau).item()
        scheduler.step(test_loss)

    return net


def plot_results(net, x_all, y_all, fname):
    order = x_all.sort(0)[1]
    x_all = x_all[order].view(-1, 1)
    y_all = y_all[order].view(-1, 1)

    plt.figure(figsize=(10, 6))
    plt.plot(x_all.numpy(), y_all.numpy(), '.')
    for tau in TAUS:
        lw = 5 if tau == 0.5 else 2
        plt.plot(x_all.numpy(),
                 net(x_all, tau).detach().numpy(),
                 label="$\\tau = " + str(tau) + "$",
                 lw=lw)

    plt.legend(ncol=3)
    plt.tight_layout(pad=0)
    plt.savefig(fname)
    plt.close('all')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantile regression')
    parser.add_argument('--root', type=str, default='../data/pairs_tuebingen')
    parser.add_argument('--pair', type=int, default=0)
    parser.add_argument('--plot', type=int, default=1)
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--n_epochs', type=int, default=3000)
    parser.add_argument('--n_hidden_units', type=int, default=50)
    parser.add_argument('--quad_int', type=int, default=3)
    parser.add_argument('--unif_int', type=int, default=0)
    args = parser.parse_args()

    global TAUS
    global wTAUS

    samples, weights, names = TuebingenDataset(args.root)
    print("Number of pairs = ", len(samples))

    x_all = qnorm_scale(samples[args.pair][:, 0])
    y_all = qnorm_scale(samples[args.pair][:, 1])

    # if the pair has less than 200 samples, casual score only from the median
    if args.quad_int == 1 or args.unif_int == 1 or x_all.shape[0] <= 200:
        TAUS = [0.5]
        wTAUS = [1]

    # Gaussian quadrature integration
    elif args.quad_int == 3:
        TAUS = [0.12, 0.5, 0.89]
        wTAUS = [0.28, 0.44, 0.28]
    elif args.quad_int == 5:
        TAUS = [0.05, 0.23, 0.5, 0.77, 0.95]
        wTAUS = [0.12, 0.24, 0.28, 0.24, 0.12]

    elif args.unif_int != 0:
        TAUS = torch.linspace(0, 1, args.unif_int)
        wTAUS = torch.Tensor([1 / args.unif_int]).repeat(args.unif_int)

    score_fw = 0
    score_bw = 0
    reps = 0

    for repetition in range(args.repetitions):
            x_tr, y_tr, x_te, y_te = train_test_split(x_all, y_all)

            net_fw = train_net(x_tr, y_tr, x_te, y_te, n_epochs=args.n_epochs)
            net_bw = train_net(y_tr, x_tr, x_te, y_te, n_epochs=args.n_epochs)

            score_fw += sum(causal_score(net_fw, x_te, y_te))
            score_bw += sum(causal_score(net_bw, y_te, x_te))

            reps += 1

    score_fw /= reps
    score_bw /= reps

    print(args.root,
          names[args.pair],
          weights[args.pair],
          score_fw,
          score_bw,
          score_fw < score_bw)

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.rc('text', usetex=True)
        plt.rc('font', size=16)

        plot_results(net_fw, x_all, y_all, names[args.pair] + "_fw.pdf")
        plot_results(net_bw, y_all, x_all, names[args.pair] + "_bw.pdf")
