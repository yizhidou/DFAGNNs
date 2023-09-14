from clrs._src import samplers
from clrs._src import algorithms
from clrs._src import yzd_utils
from clrs._src import specs, dfa_specs
from clrs._src import probing
from typing import List, Optional, Tuple
from programl.proto import *
import random
import numpy as np
import jax
import collections

Spec = specs.Spec
Features = collections.namedtuple('Features', ['input_dp_list',
                                               'trace_h',
                                               'padded_edge_indices_dict',
                                               'mask_dict'])

Feedback = collections.namedtuple('Feedback', ['features', 'trace_o'])

_Array = np.ndarray
_DataPoint = probing.DataPoint
Trajectory = List[_DataPoint]
Trajectories = List[Trajectory]


class DFASampler(samplers.Sampler):

    # 要设定max_step的
    def __init__(self,
                 task_name: str,
                 sample_id_list: List[str],
                 seed: int,
                 sample_loader: yzd_utils.SampleLoader
                 ):
        if not task_name in ['dfa_liveness', 'dfa_dominance', 'dfa_reachability']:
            raise NotImplementedError(f'No implementation of algorithm {task_name}.')
        self.sample_id_list = sample_id_list
        self.task_name = task_name
        self.sample_loader = sample_loader
        self.max_steps = self.sample_loader.max_iteration - 1
        self.max_num_pp = self.sample_loader.max_num_pp

        samplers.Sampler.__init__(self,
                                  algorithm=getattr(algorithms, task_name),
                                  spec=dfa_specs.DFASPECS[task_name],
                                  num_samples=-1,  #
                                  seed=seed,
                                  if_estimate_max_step=False)

    def _sample_data(self, length: Optional[int] = None, *args, **kwargs):
        # print(f'sample_id_list = {self.sample_id_list}')
        sample_id = random.choice(seq=self.sample_id_list)
        # print(f'randomly selected sample_id = {sample_id}')
        return sample_id

    def _make_batch(self, num_samples: int,
                    spec: Spec, min_length: int,
                    algorithm: samplers.Algorithm, *args, **kwargs):
        """Generate a batch of data."""
        inp_dp_list_list = []  # pos, if_pp, if_ip, cfg, gen, (kill,) trace_i
        outp_dp_list_list = []  # trace_o
        hint_dp_list_list = []  # trace_h
        edge_indices_dict_list = []
        mask_dict_list = []
        # nb_nodes_this_batch = np.zeros(num_samples, int)
        # nb_cfg_edges_this_batch = np.zeros(num_samples, int)
        # nb_gkt_edges_this_batch = np.zeros(num_samples, int)
        # hint_len_this_batch = np.zeros(num_samples, int)

        num_created_samples = 0
        while num_created_samples < num_samples:
            sample_id = self._sample_data(*args, **kwargs)
            try:
                edge_indices_dict, mask_dict, probes = algorithm(self.sample_loader, sample_id)
            except yzd_utils.YZDExcpetion:
                continue
            num_created_samples += 1
            edge_indices_dict_list.append(edge_indices_dict)
            mask_dict_list.append(mask_dict)
            inp_dp_list, outp_dp_list, hint_dp_list = probing.split_stages(probes, spec)
            inp_dp_list_list.append(inp_dp_list)
            outp_dp_list_list.append(outp_dp_list)
            hint_dp_list_list.append(hint_dp_list)

        # Batch and pad trajectories to max(T).
        batched_inp_dp_list = _batch_ioh(inp_dp_list_list)
        batched_trace_o = _batch_ioh(outp_dp_list_list)[0]
        batched_trace_h = _batch_ioh(hint_dp_list_list)[0]
        batched_edge_indices_dict = jax.tree_util.tree_map(lambda *x: np.stack(x), *edge_indices_dict_list)
        batched_mask_dict = jax.tree_util.tree_map(lambda *x: np.array(mask_dict), *mask_dict_list)
        return batched_edge_indices_dict, batched_mask_dict, batched_inp_dp_list, batched_trace_o, batched_trace_h

    def next(self, batch_size: Optional[int] = None) -> Feedback:
        """Subsamples trajectories from the pre-generated dataset.

        Args:
          batch_size: Optional batch size. If `None`, returns entire dataset.

        Returns:
          Subsampled trajectories.
        """
        if not batch_size:
            # YZDTODO should raise an error
            batch_size = 1
        batched_edge_indices_dict, batched_mask_dict, batched_inp_dp_list, batched_trace_o, batched_trace_h = self._make_batch(
            num_samples=batch_size,
            spec=self._spec,
            min_length=self.max_steps,
            algorithm=self._algorithm)
        # assert np.array_equal(lengths, sparse_lengths)
        assert len(batched_inp_dp_list) == 7 if self.task_name == 'dfa_liveness' else 6

        return Feedback(features=Features(input_dp_list=batched_inp_dp_list,
                                          trace_h=batched_trace_h,
                                          padded_edge_indices_dict=batched_edge_indices_dict,
                                          mask_dict=batched_mask_dict),
                        trace_o=batched_trace_o)


def build_dfa_sampler(task_name: str,
                      sample_id_list: List[str],
                      seed: int,
                      sample_loader: yzd_utils.SampleLoader
                      ) -> Tuple[DFASampler, Spec]:
    """Builds a sampler. See `Sampler` documentation."""

    if task_name not in ['dfa_liveness', 'dfa_dominance', 'dfa_reachability']:
        raise NotImplementedError(f'No implementation of algorithm {task_name}.')
    spec = dfa_specs.DFASPECS[task_name]

    sampler = DFASampler(task_name=task_name,
                         sample_id_list=sample_id_list,
                         seed=seed,
                         sample_loader=sample_loader)
    return sampler, spec


def _batch_ioh(ioh_dp_list_list: Trajectories) -> Trajectory:
    assert ioh_dp_list_list
    for sample_idx, dp_list_one_sample in enumerate(ioh_dp_list_list):
        for dp_idx, dp in enumerate(dp_list_one_sample):
            if dp.name == 'trace_h':
                assert dp.data.shape[1] == 1
                concat_dim = 1
            else:
                assert dp.data.shape[0] == 1
                concat_dim = 0
    return jax.tree_util.tree_map(lambda *x: np.concatenate(x,
                                                            axis=concat_dim),
                                  *ioh_dp_list_list)
