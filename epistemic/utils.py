# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
from torchvision import datasets, transforms
from torch.nn import functional as F
from torch.autograd import grad
import torch

from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import cdist

import numpy as np

from resnet import PreActResNet


def uncertainty_oracle(network, loader_in, loader_out, n_iterations=10000):
    f_in = featurize_loader(network, loader_in)
    f_out = featurize_loader(network, loader_out)

    n = min(f_in.size(0), f_out.size(0))
    perm = torch.randperm(n)

    inputs = torch.cat((f_in[perm[:n]], f_out[perm[:n]]))
    outputs = torch.cat((torch.zeros(n, 1), torch.ones(n, 1)))

    oracle = torch.nn.Linear(f_in.size(1), 1)
    opt = torch.optim.Adam(oracle.parameters())
    loss = torch.nn.BCEWithLogitsLoss()

    for iteration in range(n_iterations):
        opt.zero_grad()
        loss(oracle(inputs), outputs).backward()
        opt.step()

    def handle(network, loader, args=None):
        f = featurize_loader(network, loader)
        return oracle(f).detach()

    return handle


def uncertainty_covariance(network, loader, lamba=1):
    """
    Compute uncertainty by fitting a Gaussian to the last layer representation
    """
    f = featurize_loader(network, loader)
    icov = f.t().mm(f).add(torch.eye(f.size(1)) * lamba).inverse()

    def handle(network, loader):
        f = featurize_loader(network, loader)
        return f.mm(icov).mm(f.t()).diag().view(-1).detach()

    return handle


def uncertainty_distance(network, loader_out, loader_in, percentile=0):
    """
    Compute uncertainty using the distance to nearest training point at the
    last layer representation
    """
    f_in = featurize_loader(network, loader_in)
    f_out = featurize_loader(network, loader_out)
    dist = cdist(f_in.numpy(), f_out.numpy())

    return torch.Tensor(np.percentile(dist, percentile, 0))


def uncertainty_sphere(network, loader):
    """
    Compute uncertainty as distance from sphere 
    """
    features = featurize_loader(network, loader)
    return (features.norm(2, 1) - 1).pow(2).detach()


def uncertainty_entropy(network, loader, temperature=1):
    """
    Compute uncertainty using softmax entropy
    """
    outputs = featurize_loader(network,
                               loader,
                               classify=True,
                               logits=False,
                               temperature=temperature).add(1e-3)

    return (outputs * outputs.log()).sum(1).mul(-1).detach()


def uncertainty_largest(network, loader, logits=0, temperature=1):
    """
    Compute uncertainty as the negative largest logit 
    """
    outputs = featurize_loader(network,
                               loader,
                               classify=True,
                               logits=logits,
                               temperature=temperature)

    return outputs.max(1)[0].mul(-1)


def uncertainty_functional_margin(network, loader, logits=0, temperature=1):
    """
    Compute uncertainty as the difference between the two largest logits
    """
    outputs = featurize_loader(network,
                               loader,
                               classify=True,
                               logits=logits,
                               temperature=temperature)

    top = outputs.topk(2)[0]
    return (top[:, 0] - top[:, 1]).mul(-1)


def uncertainty_geometrical_margin(network, loader, logits=0, temperature=1):
    """
    Compute uncertainty as the (linearized) distance to the decision boundary
    """
    m = []

    for (x, _) in loader:
        x.requires_grad = True
        predictions = network(x)
        top = predictions
        if logits == 0:
            top = F.softmax(top.div(temperature), dim=1)
        top = top.topk(2)[0]
        functional = top[:, 0] - top[:, 1]
        geometrical = grad(predictions.sum(), x)[
            0].view(x.size(0), -1).norm(2, 1)
        m.append((functional.cpu() / geometrical.cpu()).mul(-1).detach())

    return torch.cat(m).squeeze()


def uncertainty_distillation(network, loader, certs_k=100, certs_epochs=10):
    """
    Compute uncertainty using linear random network distillation 
    """
    features = featurize_loader(network, loader)

    labeler = torch.nn.Sequential(
        torch.nn.Linear(features.size(1), 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, certs_k))

    targets = labeler(features).detach()

    certificates = torch.nn.Linear(features.size(1), certs_k)
    opt = torch.optim.Adam(certificates.parameters())

    loss = torch.nn.MSELoss()

    loader_f = DataLoader(TensorDataset(features, targets),
                          batch_size=64,
                          shuffle=True)

    for epoch in range(certs_epochs):
        for f, t in loader_f:
            opt.zero_grad()
            loss(certificates(f), t).backward()
            opt.step()

    def handle(network, loader_test, args=None):
        features_test = featurize_loader(network, loader_test)
        return (certificates(features_test) - labeler(features_test).detach()).pow(2).mean(1).detach()

    return handle


def uncertainty_pca(network, loader, k=100, where="top"):
    """
    Compute uncertainty using PCA 
    """
    features = featurize_loader(network, loader)
    mean = features.mean(0, keepdim=True)

    U, S, V = torch.svd(features - mean)

    def handle(network, loader_test, args=None):
        features_test = featurize_loader(network, loader_test)
        if where == "top":
            return ((features_test - mean) @ V[:, :k]).mean(1).mul(-1).detach()
        elif where == "bottom":
            return ((features_test - mean) @ V[:, -k:]).mean(1).detach()

    return handle


def uncertainty_certificates(network,
                             loader,
                             certs_loss="bce",
                             certs_k=100,
                             certs_epochs=10,
                             certs_reg=0,
                             certs_bias=0):
    """
    Compute uncertainty using linear certificates (ours)
    """
    features = featurize_loader(network, loader)
    
    def target(x):
        return torch.zeros(x.size(0), certs_k)

    certificates = torch.nn.Linear(features.size(1), certs_k, bias=certs_bias)
    opt = torch.optim.Adam(certificates.parameters())
    sig = torch.nn.Sigmoid()

    if certs_loss == "bce":
        loss = torch.nn.BCEWithLogitsLoss(reduction="none")
    elif certs_loss == "mse":
        loss = torch.nn.L1Loss(reduction="none")

    loader_f = DataLoader(TensorDataset(features, features),
                          batch_size=64,
                          shuffle=True)

    for epoch in range(certs_epochs):
        for f, _ in loader_f:
            opt.zero_grad()
            error = loss(certificates(f), target(f)).mean()
            penalty = certs_reg * \
                (certificates.weight @ certificates.weight.t() - 
                 torch.eye(certs_k)).pow(2).mean()
            (error + penalty).backward()
            opt.step()

    def handle(network, loader_test, args=None):
        f = featurize_loader(network, loader_test)
        output = certificates(f)
        if certs_loss == "bce":
            return sig(output).pow(2).mean(1).detach()
        else:
            return output.pow(2).mean(1).detach()

    return handle


def uncertainty_svdd(network, loader, svdd_k=100, svdd_epochs=10):
    """
    Instantiate an SVDD model
    """
    num_channels = loader.dataset[0][0].size(0)
    svdd = PreActResNet(num_channels=num_channels, num_classes=svdd_k, bias=False)

    outputs = featurize_loader(svdd, loader, classify=True, logits=1)
    target = outputs.mean(0, keepdim=True).detach().to(svdd.device)

    optimizer = torch.optim.Adam(svdd.parameters())

    for epoch in range(svdd_epochs):
        for (xi, _) in loader:
            optimizer.zero_grad()
            (svdd(xi) - target).pow(2).mean().backward()
            optimizer.step()

    svdd.eval()
    target = target.cpu()

    def handle(network, loader_test, args=None):
        outputs = featurize_loader(svdd, loader_test, classify=True, logits=1)
        return (outputs - target).pow(2).mean(1)

    return handle


def uncertainty_odin(network, loader, temperature=10, eps=0.1):
    """
    Compute uncertainty using ODIN
    """
    all_scores = []

    cross_entropy = torch.nn.CrossEntropyLoss()

    for (image, _) in loader:
        image.requires_grad = True

        predicted_logits = network(image) / temperature
        predicted_labels = predicted_logits.argmax(1).detach()

        cross_entropy(predicted_logits, predicted_labels).backward()

        perturbed_image = image.add(image.grad.mul(-1).sign().mul(-eps))

        scores = F.softmax(network(perturbed_image) / temperature, dim=1)
        scores = scores.max(1)[0].detach()

        all_scores.append(scores.cpu())

    return torch.cat(all_scores).view(-1).detach()


def uncertainty_random(network, loader):
    return torch.rand(len(loader.dataset), 1)


def load_uncertainty_measure(network, loader_tr, loader_te, args):
    if args.method == "covariance":
        method = uncertainty_covariance(network,
                                        loader_tr,
                                        args.cov_lamba)
        uncertainty_args = {}
    elif args.method == "distance":
        method = uncertainty_distance
        uncertainty_args = {'loader_in': loader_tr,
                            'percentile': args.dist_percentile}
    elif args.method == "largest":
        method = uncertainty_largest
        uncertainty_args = {'logits': args.logits,
                            'temperature': args.temperature}
    elif args.method == "entropy":
        method = uncertainty_entropy
        uncertainty_args = {'temperature': args.temperature}
    elif args.method == "functional":
        method = uncertainty_functional_margin
        uncertainty_args = {'logits': args.logits,
                            'temperature': args.temperature}
    elif args.method == "geometrical":
        method = uncertainty_geometrical_margin
        uncertainty_args = {'logits': args.logits,
                            'temperature': args.temperature}
    elif args.method == "odin":
        method = uncertainty_odin
        uncertainty_args = {'temperature': args.temperature,
                            'eps': args.eps}
    elif args.method == "certificates":
        method = uncertainty_certificates(network,
                                          loader_tr,
                                          args.certs_loss,
                                          args.certs_k,
                                          args.certs_epochs,
                                          args.certs_reg,
                                          args.certs_bias)
        uncertainty_args = {}
    elif args.method == "distillation":
        method = uncertainty_distillation(network,
                                          loader_tr,
                                          args.certs_k,
                                          args.certs_epochs)
        uncertainty_args = {}
    elif args.method == "oracle":
        method = uncertainty_oracle(network,
                                    loader_tr,
                                    loader_te)
        uncertainty_args = {}
    elif args.method == "random":
        method = uncertainty_random
        uncertainty_args = {}
    elif args.method == "sphere":
        method = uncertainty_sphere
        uncertainty_args = {}
    elif args.method == "pca":
        method = uncertainty_pca(network,
                                 loader_tr,
                                 args.certs_k,
                                 args.where)
        uncertainty_args = {}
    elif args.method == "svdd":
        method = uncertainty_svdd(network,
                                  loader_tr,
                                  args.certs_k,
                                  args.certs_epochs)
        uncertainty_args = {}
    else:
        raise NotImplementedError

    def handle(loader):
        return method(network, loader, **uncertainty_args)

    return handle


def featurize_loader(network, loader, classify=False, logits=1, temperature=1):
    softmax = torch.nn.Softmax(dim=1)
    features = []

    for (x, _) in loader:
        if classify:
            predictions = network(x)
            if logits == 0:
                predictions = softmax(predictions.div(temperature))
            features.append(predictions.detach())
        else:
            features.append(network.features(x.to(network.device)).detach())

    return torch.cat(features).squeeze().cpu()


def classification_accuracy(network, loader):
    p = []
    t = []
    for (x, y) in loader:
        p.append(network(x).detach())
        t.append(y.long().view(-1))

    p = torch.cat(p).squeeze()
    t = torch.cat(t).squeeze()
    acc = (p.cpu().argmax(1) == t).float().mean().item()

    if len(t.unique()) == 2:
        p = F.softmax(p, dim=1)[:, 1]
        aucroc = roc_auc_score(t.cpu().numpy(), p.cpu().numpy())
    else:
        aucroc = 0

    return acc, aucroc


def load_dataset(name="cifar",
                 root="../data/",
                 categories="all",
                 train=True,
                 augment=False,
                 batch_size=64):
    transform = []
    transform.append(transforms.Resize(32))

    if augment:
        transform.append(transforms.RandomCrop(32, padding=4))
        transform.append(transforms.RandomHorizontalFlip())

    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize((0.5, 0.5, 0.5),
                                          (0.5, 0.5, 0.5)))
    transform = transforms.Compose(transform)

    if name == "cifar":
        the_dataset = datasets.CIFAR10
    elif name == "mnist":
        the_dataset = datasets.MNIST
    elif name == "fashion":
        the_dataset = datasets.FashionMNIST

    if name == "svhn":
        dataset = datasets.SVHN(root,
                                transform=transform,
                                download=True,
                                split="train" if train else "test")
    else:
        dataset = the_dataset(root,
                              transform=transform,
                              download=True,
                              train=train)

    if categories != "all":
        accepted = []

        for i in range(len(dataset)):
            image, label = dataset[i]

            if label in categories:
                accepted.append(i)

    c = categories
    dataset.target_transform = lambda x: c.index(x) if x in c else None
    
    subdataset = Subset(dataset, accepted)

    return DataLoader(subdataset,
                      shuffle=True,
                      num_workers=8,
                      pin_memory=True,
                      batch_size=batch_size)
