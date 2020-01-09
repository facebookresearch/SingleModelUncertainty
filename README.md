# Single-Model Uncertainties for Deep Learning

Source code for "[Single-Model Uncertainties for Deep
Learning](https://arxiv.org/abs/1811.00908)", by Natasa Tagasovska and David
Lopez-Paz, NeurIPS 2019.

Contents:
```
├── README.txt (this file)
├── toy (script to generate Figure 1)
├── aleatoric (all experiments related to the aleatoric estimator)
│   └──regression (Section 4.1)
│       ├── regression_experiment.py
│       └── regression_experiment_sweep.json
│	└── joint_estimation.py (Figure 2)
│	└──regression_experiment_parse_table
│   └──causality 
│       ├── causal_discovery.py (A.1)
│       └── heterogeneous_qte.py (A.2)
├── epistemic (all experiments related to the epistemic estimator)
│   ├── image_experiment.py (Section 4.2)
│   ├── image_experiment_sweep.py
│   ├── resnet.py
│   └── utils.py
└── data
    ├── UCI_Datasets
    ├── kdg.csv
    ├── pairs_an
    ├── pairs_ls
    ├── pairs_ls-s
    ├── pairs_mn-u
    └── pairs_tuebingen
```

The image datasets will be downloaded once the scripts are run.

This source code is released under a Attribution-NonCommercial 4.0 International
license, find out more about it [here](LICENSE).
