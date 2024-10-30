# DFA-GNNs

GNN models aligned with the Data-flow analysis (DFA) algorithm. 

<img src="architecture_formula.pdf" alt="Alt text" style="zoom:60%;" />

Please note that this project is developed based on [The CLRS Algorithmic Reasoning Benchmark](https://github.com/google-deepmind/clrs) . Therefore their original code is also kept. All the code for our experiments is under the dfa/ directory. Please check `dfa_processors.py` for the specific implementations of DFA-GNN+, DFA-GNN, and DFA-GNN-.

- Dataset
  - We employ the dataset from [ProGraML](https://github.com/ChrisCummins/ProGraML) project.

- Training
  - Basically you just need to set up the training params according to `demo_train_params.txt`, and run `python dfa_train.py --params /demo_train_params.txt`dfa_train.py.
- Validation/Testing
  - Set up the validation/testing params according to `demo_test_param.txt` and run `python dfa_vali_test.py --params demo_test_param.txt`



