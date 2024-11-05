# DFA-GNNs

**This documentation is on updating...**

GNN models aligned with the Data-flow analysis (DFA) algorithm. 

Please note that this project is developed based on [The CLRS Algorithmic Reasoning Benchmark](https://github.com/google-deepmind/clrs) . Therefore their original code is also kept. All the code for our experiments is under the dfa/ directory. Please check `dfa_processors.py` for the specific implementations of DFA-GNN+, DFA-GNN, and DFA-GNN-.

- Prerequisites
  - To replicate the experiments from the paper,  you need to install our edited version of [ProGraML](https://anonymous.4open.science/r/programl_edited-DBDB/README.md) as a Python package.

- Dataset
  - We employ the dataset from [ProGraML Dataset](https://anonymous.4open.science/r/programl_edited-DBDB/README.md). Please note that only Control-flow graphs (CFGs) are used, ground-truth labels should be generated from edited version of [ProGraML](https://anonymous.4open.science/r/programl_edited-DBDB/README.md) project.

- Training
  - Basically you just need to set up the training params according to `demo_train_params.txt`, and run `python dfa_train.py --params /demo_train_params.txt`dfa_train.py.
- Validation/Testing
  - Set up the validation/testing params according to `demo_test_param.txt` and run `python dfa_vali_test.py --params demo_test_param.txt`

## Settings for each experiment

### Experiment I





