# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.:wq

import os
import torch
import argparse

import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegressionCV

from utils import create_adversarial_examples
from utils import load_uncertainty_measure
from utils import classification_accuracy
from utils import featurize_loader
from utils import load_dataset
from utils import VGG

from resnet import PreActResNet


def compute_uncertainty(network,
                        loader_in_tr,
                        loader_in_te,
                        loader_out_te,
                        args):
    uncertainty_measure = load_uncertainty_measure(network,
                                                   loader_in_tr,
                                                   loader_out_te,  # for oracle
                                                   args)

    uscores_in_tr = uncertainty_measure(loader_in_tr)
    uscores_in_te = uncertainty_measure(loader_in_te)
    uscores_out_te = uncertainty_measure(loader_out_te)

    thres_90, thres_95, thres_99 = np.percentile(uscores_in_tr.view(-1).numpy(),
                                                 (90, 95, 99))

    uscores = torch.cat((uscores_in_te, uscores_out_te)).numpy()
    ulabels = torch.cat((torch.zeros(len(uscores_in_te)),
                         torch.ones(len(uscores_out_te)))).numpy()

    acc_lr = 0
    for t in np.linspace(uscores.min(), uscores.max(), 1000):
        acc_t = ((uscores > t) == ulabels).mean()
        if acc_t > acc_lr:
            acc_lr = acc_t

    acc_90 = ((uscores > thres_90) == ulabels).mean()
    acc_95 = ((uscores > thres_95) == ulabels).mean()
    acc_99 = ((uscores > thres_99) == ulabels).mean()
    auc = roc_auc_score(ulabels, uscores)

    return auc, acc_lr, acc_90, acc_95, acc_99


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Out-of-distribution image experiment")
    parser.add_argument('--dataset', type=str, default="cifar")
    parser.add_argument('--adversarial', type=int, default=0)
    parser.add_argument('--adv_eps', type=float, default=.1)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--method', type=str, default="bald")
    parser.add_argument('--cov_lamba', type=float, default=0.01)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--dist_percentile', type=float, default=1)
    parser.add_argument('--network', type=str, default="")
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--certs_k', type=int, default=100)
    parser.add_argument('--certs_epochs', type=int, default=10)
    parser.add_argument('--certs_loss', type=str, default="bce")
    parser.add_argument('--certs_reg', type=float, default=0)
    parser.add_argument('--certs_bias', type=int, default=0)
    parser.add_argument('--logits', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_drop', type=int, default=10)
    parser.add_argument('--where', type=str, default="top")
    parser.add_argument('--network_type', type=str, default="resnet")
    parser.add_argument('--unit_features', type=int, default=0)
    args = parser.parse_args()

    # set up seeds ############################################################

    torch.backends.cudnn.benchmark = True

    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load data ###############################################################

    if args.network == "":
        categories = torch.randperm(10).tolist()
        dataset = args.dataset
    else:
        categories = torch.load(args.network)[1]
        dataset = os.path.basename(args.network).split("_")[1]
        args.dataset = dataset

    loader_in_tr = load_dataset(name=dataset,
                                categories=categories[:5],
                                train=True,
                                augment=(args.network == ""))
    loader_in_te = load_dataset(name=dataset,
                                categories=categories[:5],
                                train=False)
    loader_out_te = load_dataset(name=dataset,
                                 categories=categories[5:],
                                 train=False)

    args.n_classes = 5

    # train or load network ###################################################

    if dataset == "mnist" or dataset == "fashion":
        num_channels = 1
    else:
        num_channels = 3

    if args.network_type == "VGG":
        network = VGG(num_classes=5, dropout=(args.method == "bald"))
    else:
        network = PreActResNet(num_channels=num_channels, num_classes=5)

    network.train()

    if args.network == "":
        optimizer = torch.optim.Adam(network.parameters())
        loss = torch.nn.CrossEntropyLoss()
        loss.to(network.device)

        for epoch in range(args.num_epochs):
            epoch_error = 0
            pbar = tqdm(loader_in_tr, ncols=80, leave=False)
            for i, (x, y) in enumerate(pbar):
                optimizer.zero_grad()
                features = network.features(x)
                outputs = network.classify(features)
                error = loss(outputs, y.to(network.device))

                if args.unit_features:
                    penalty = (features.norm(2, 1) - 1).pow(2).mean()
                else:
                    penalty = 0

                (error + penalty).backward()
                optimizer.step()

                epoch_error += error.item()
                pbar.set_description("{:.4f}".format(epoch_error / (i + 1)))
               
            if (epoch + 1) % 10 == 0:
                acc_te, _ = classification_accuracy(network, loader_in_te)
                print(epoch + 1, acc_te)

        torch.save((network.state_dict(), categories, args.unit_features),
                   "network_" +
                   dataset +
                   "_" +
                   '_'.join(str(x) for x in categories) +
                   '_unit=' + str(args.unit_features) +
                   ".pt")
    elif args.network == "random":
        pass
    else:
        network.load_state_dict(torch.load(args.network)[0])

    network.eval()

    # adversarial out-of-distribution #########################################

    if args.adversarial == 1:
        loader_out_te = create_adversarial_examples(network,
                                                    loader_in_te,
                                                    adv_eps=args.adv_eps)

    # run out-of-distribution detection #######################################

    uncertainty_stats = compute_uncertainty(network,
                                            loader_in_tr,
                                            loader_in_te,
                                            loader_out_te,
                                            args)

    print("{:<12} {:<12} {} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} # {}".format(
          dataset,
          args.method,
          args.seed,
          *uncertainty_stats,
          str(vars(args))))
