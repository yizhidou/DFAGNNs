# DFA-GNNs

**This documentation is on updating...**

GNN models aligned with the Data-flow analysis (DFA) algorithm. 

Please note that this project is developed based on [The CLRS Algorithmic Reasoning Benchmark](https://github.com/google-deepmind/clrs) . Therefore their original code is also kept. All the code for our experiments is under the dfa/ directory. Please check `dfa_processors.py` for the specific implementations of DFA-GNN+, DFA-GNN, and DFA-GNN-.

- **Prerequisites**
  - To replicate the experiments from the paper,  you need to install our edited version of [ProGraML](https://anonymous.4open.science/r/programl_edited-DBDB/README.md) as a Python package.

- **Datasets**
  - We employ the dataset from [ProGraML Dataset](https://anonymous.4open.science/r/programl_edited-DBDB/README.md). Please note that only Control-flow graphs (CFGs) are used, ground-truth labels should be generated from edited version of [ProGraML](https://anonymous.4open.science/r/programl_edited-DBDB/README.md) project.

- **Training Command**
  - Set up the training params according to `demo_train_params.json`, and run `python dfa_train.py --params /demo_train_params.json`.
- **Validation/Testing Command**
  - Set up the validation/testing params according to `demo_test_param.json` and run `python dfa_vali_test.py --params demo_test_param.json`

## Training Setting Files

### Experiment I

Train models under two modes (with or without trajectory supervision).

All settings of this experiment are under directory `training_settings/experiment_1/`

- **w/o trajectory supervision**
  - DFA-GNN$^+$
    -   training arguments: `without_trajectory/plus_wo.json`
  - DFA-GNN
    - training arguments: `without_trajectory/gnn_wo.json`
  - DFA-GNN$^-$
    - training arguments: `without_trajectory/minus_wo.json`
- **with trajectory supervision**
  - DFA-GNN$^+$
    -   training arguments: `with_trajectory/plus_w.json`
  - DFA-GNN
    - training arguments: `with_trajectory/gnn_w.json`
  - DFA-GNN$^-$
    - training arguments: `with_trajectory/minus_w.json`

### Experiment II

Train models with various trajectory-length (50, 10, 5, 1).

All settings of this experiment are under directory `training_settings/experiment_2/`

- `trajectory_len = 50`
  - DFA-GNN$^+$
    -   training arguments: `trace_len_50/plus_50.json`
  - DFA-GNN
    - training arguments: `trace_len_50/gnn_50.json`
  - DFA-GNN$^-$
    - training arguments: `trace_len_50/minus_50.json`
- `trajectory_len = 10`
  - DFA-GNN$^+$
    -   training arguments: `trace_len_10/plus_10.json`
  - DFA-GNN
    - training arguments: `trace_len_10/gnn_10.json`
  - DFA-GNN$^-$
    - training arguments: `trace_len_10/minus_10.json`
- `trajectory_len = 5`
  - DFA-GNN$^+$
    -   training arguments: `trace_len_5/plus_5.json`
  - DFA-GNN
    - training arguments: `trace_len_5/gnn_5.json`
  - DFA-GNN$^-$
    - training arguments: `trace_len_5/minus_5.json`
- `trajectory_len = 1`
  - DFA-GNN$^+$
    -   training arguments: `trace_len_1/plus_1.json`
  - DFA-GNN
    - training arguments: `trace_len_1/gnn_1_full.json`
  - DFA-GNN$^-$
    - training arguments: `trace_len_1/minus_1_full.json`

### Experiment III

Train models with various numbers of samples (1, 10, 100, 1000, full)

All settings of this experiment are under directory `training_settings/experiment_3/`

To generate means and standard deviations of the model performances, each training set with 1, 10, 100, or 1000 samples has 3 different versions, corresponding to `v1`, `v2`, and `v3` settings. For the full training set, we change the random seeds instead.

- `train_set_size = 1`
  - DFA-GNN$^+$
    -   training arguments: `train_set_size_1/plus_1_v1.json`, `train_set_size_1/plus_1_v2.json`, `train_set_size_1/plus_1_v3.json`
  - DFA-GNN
    - training arguments: `train_set_size_1/gnn_1_v1.json`, `train_set_size_1/gnn_1_v2.json`, `train_set_size_1/gnn_1_v3.json`
  - DFA-GNN$^-$
    - training arguments: `train_set_size_1/minus_1_v1.json`, `train_set_size_1/minus_1_v2.json`, `train_set_size_1/minus_1_v3.json`
- `train_set_size = 10`
  - DFA-GNN$^+$
    -   training arguments: `train_set_size_10/plus_10_v1.json`, `train_set_size_10/plus_10_v2.json`, `train_set_size_10/plus_10_v3.json`
  - DFA-GNN
    - training arguments: `train_set_size_10/gnn_10_v1.json`, `train_set_size_10/gnn_10_v2.json`, `train_set_size_10/gnn_10_v3.json`
  - DFA-GNN$^-$
    - training arguments: `train_set_size_10/minus_10_v1.json`, `train_set_size_10/minus_10_v2.json`, `train_set_size_10/minus_10_v3.json`
- `train_set_size = 100`
  - DFA-GNN$^+$
    -   training arguments: `train_set_size_100/plus_100_v1.json`, `train_set_size_100/plus_100_v2.json`, `train_set_size_100/plus_100_v3.json` 
  - DFA-GNN
    - training arguments: `train_set_size_100/gnn_100_v1.json`, `train_set_size_100/gnn_100_v2.json`, `train_set_size_100/gnn_100_v3.json`
  - DFA-GNN$^-$
    - training arguments: `train_set_size_100/minus_100_v1.json`, `train_set_size_100/minus_100_v2.json`, `train_set_size_100/minus_100_v3.json`
- `train_set_size = 1000`
  - DFA-GNN$^+$
    -   training arguments: `train_set_size_1000/plus_1000_v1.json`, `train_set_size_1000/plus_1000_v2.json`, `train_set_size_1000/plus_1000_v3.json`
  - DFA-GNN
    - training arguments: `train_set_size_1000/gnn_1000_v1.json`, `train_set_size_1000/gnn_1000_v2.json`, `train_set_size_1000/gnn_1000_v3.json`
  - DFA-GNN$^-$
    - training arguments: `train_set_size_1000/minus_1000_v1.json`, `train_set_size_1000/minus_1000_v2.json`, `train_set_size_1000/minus_1000_v3.json`
- `train_set_size = 'full'`
  - DFA-GNN$^+$
    -   training arguments: `train_set_size_full/plus_full_v1.json`, `train_set_size_full/plus_full_v2.json`, `etrain_set_size_full/plus_full_v3.json`
  - DFA-GNN
    - training arguments: `train_set_size_full/gnn_full_v1.json`, `train_set_size_full/gnn_full_v2.json`, `train_set_size_full/gnn_full_v3.json`
  - DFA-GNN$^-$
    - training arguments: `train_set_size_full/minus_full_v1.json`, `train_set_size_full/minus_full_v2.json`, `train_set_size_full/minus_full_v3.json`

### Experiment IV

Train models on various projects (POJ104, GitHub, Linux, TensorFlow)

All settings of this experiment are under directory `training_settings/experiment_4/`

- POJ104
  - DFA-GNN$^+$
    - `poj104/plus_poj104.json`
  - DFA-GNN
    - `poj104/gnn_poj104.json`
  - DFA-GNN$^-$
    - `poj104/minus_poj104.json`
- GitHub
  - DFA-GNN$^+$
    - `github/plus_github.json`
  - DFA-GNN
    - `github/gnn_github.json`
  - DFA-GNN$^-$
    - `github/minus_github.json`
- Linux
  - DFA-GNN$^+$
    - `linux/plus_linux.json`
  - DFA-GNN
    - `linux/gnn_linux.json`
  - DFA-GNN$^-$
    - `linux/minus_linux.json`
- TensorFlow
  - DFA-GNN$^+$
    - `tensorflow/plus_tensorflow.json`
  - DFA-GNN
    - `tensorflow/gnn_tensorflow.json`
  - DFA-GNN$^-$
    - `tensorflow/minus_tensorflow.json`

