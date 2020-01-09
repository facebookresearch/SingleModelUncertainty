# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product


def simple_sweep(grid, prefix=""):
    permutations = product(*grid.values())
    result = []

    for permutation in permutations:
        string = prefix + " "
        for i, key in enumerate(grid.keys()):
            string += "--{} {} ".format(key, permutation[i])
        result.append(string)

    return result


def sweep():
    parameters = {
        "network": [
	    "../models/network_cifar_4_1_7_5_3_9_0_8_6_2.pt",
	    "../models/network_fashion_4_1_7_5_3_9_0_8_6_2.pt",
	    "../models/network_mnist_4_1_7_5_3_9_0_8_6_2.pt",
            "../models/network_svhn_4_1_7_5_3_9_0_8_6_2.pt",
        ],
        "method": {
            "oracle": {},
            "random": {},
            "covariance": {
                "cov_lamba": [1e-5, 1e-3, 1e-1, 1]
            },
            "distance": {
                "dist_percentile": [0, 1, 10, 50]
            },
            "certificates": {
                "certs_k": [100],
                "certs_epochs": [10],
                "certs_loss": ["bce"],
                "certs_reg": [0, 1, 10],
            },
            "distillation": {
                "certs_k": [100, 1000],
                "certs_epochs": [10, 100]
            },
            "entropy": {
                "temperature": [1, 2, 10]
            },
            "functional": {
                "logits": [0, 1],
                "temperature": [1, 2, 10]
            },
            "geometrical": {
                "logits": [0, 1],
                "temperature": [1, 2, 10]
            },
            "largest": {
                "logits": [0, 1],
                "temperature": [1, 2, 10]
            },
            "odin": {
                "eps": [0.014, 0.0014, 0.00014],
                "temperature": [100, 1000, 10000]
            },
            "svdd": {
                "certs_k": [100, 1000],
                "certs_epochs": [10, 100]
            },
            "pca": {
                "certs_k": [1, 10, 100],
                "where": ["bottom", "top"]
            },
            "bald": {
                "n_drop": [100]
            }
        }
    }

    commands = []

    for network in parameters["network"]:
        for method in parameters["method"]:
            prefix = "--network {} --method {}".format(network, method)
            if len(parameters["method"][method]) == 0:
                commands.append(prefix)
            else:
                commands += simple_sweep(parameters["method"][method], prefix)

    return commands

