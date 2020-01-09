# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--file', type=str, default='results_regression_all.txt')
    args = parser.parse_args()

    optimize_picp = True 

    f = open(args.file, "r")
    lines = f.readlines()
    f.close()

    results = {}
    all_nets = set()

    for line in lines:
        line_fields = [sj.split(" ") for sj in [si.rstrip().lstrip()
                                                for si in line.split("|")]]
        line_fields = [item for sublist in line_fields for item in sublist]

        net = line_fields[0]
        dataset = line_fields[1]

        objective_va = float(line_fields[5])
        captured_va = float(line_fields[6])
        width_va = float(line_fields[7])

        captured_te = float(line_fields[9])
        width_te = float(line_fields[10])

        seed = int(line_fields[11])

        if dataset not in results:
            results[dataset] = {}

        if net not in results[dataset]:
            results[dataset][net] = {}

        all_nets.add(net)

        if seed not in results[dataset][net]:
            results[dataset][net][seed] = []

        results[dataset][net][seed].append([objective_va,
                                            captured_va,
                                            width_va,
                                            captured_te,
                                            width_te])

    all_nets = list(all_nets)
    all_nets.sort()
    all_nets.remove("ConditionalQuantile-1")
    all_nets.remove("RandomForestQR-1")
    # all_nets.remove("GradientBoostingQR-1")
    all_nets.append("ConditionalQuantile-1")
    results.pop("year-song")
    results.pop("protein-tertiary-structure")

    for dataset in results:
        for method in all_nets:
            test_mpiws = []
            test_picps = []
            if method in results[dataset]:
                for seed in results[dataset][method]:
                    r = np.array(results[dataset][method][seed])
                    j = np.intersect1d(np.where(r[:, 1] > 0.925)[0], np.where(r[:, 1] < 0.975)[0])
                    if len(j):
                        rj = r[j]
                        k = np.argmin(rj[:, 2])
                        test_picps.append(rj[k, 3])
                        test_mpiws.append(rj[k, 4])

                #test_picps = np.array(test_picps).ravel()
                #test_mpiws = np.array(test_mpiws).ravel()

                if test_picps:
                    results[dataset][method]["summary"] = "${:.2f} \\pm {:.2f}$ $({:.2f} \\pm{:.2f})$".format(np.mean(test_picps), np.std(test_picps), np.mean(test_mpiws), np.std(test_mpiws))
                else:
                    results[dataset][method]["summary"] = "none"
                #results[dataset][method]["summary"] = "${:.2f} \\pm {:.2f}$ & {{\color{{gray}} ${:.2f} \\pm {:.2f}$}}".format(
                #        np.mean(test_primary),
                #        np.std(test_primary),
                #        np.mean(test_secondary),
                #        np.std(test_secondary))

    print("\\resizebox{\\textwidth}{!}{")
    print("\\begin{tabular}{lcccccccccccccc}")
    print("\\toprule")
    if optimize_picp:
        row_str = "{:<30} & ".format("\\textbf{PICP}")
    else:
        row_str = "{:<30} & ".format("\\textbf{MPIW}")

    for method in all_nets:
        row_str += "{:<32}".format(method[:-2] + " & ")
    row_str = row_str[:-3]
    row_str += "\\\\"
    print(row_str)
    print("\\midrule")

    for dataset in results:
        if "-" in dataset:
            dname = dataset.split("-")[0]
        elif dataset == "bostonHousing":
            dname = "boston"
        else:
            dname = dataset

        row_str = "{:<30}".format(dname) + " & "
        for method in all_nets:
            if method in results[dataset]:
                row_str += results[dataset][method]["summary"] + " & "
            else:
                row_str += "                                 & "
        print(row_str[:-2] + "\\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("}")
