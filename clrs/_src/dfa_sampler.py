from clrs._src import samplers
from clrs._src import algorithms
from clrs._src import dfa_utils
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
                 sample_loader: dfa_utils.SampleLoader
                 ):
        if not task_name in ['dfa_liveness', 'dfa_dominance', 'dfa_reachability']:
            raise NotImplementedError(f'No implementation of algorithm {task_name}.')
        self.sample_id_list = sample_id_list
        self.task_name = task_name
        self.sample_loader = sample_loader
        self.expected_hint_len = self.sample_loader.expected_hint_len
        self.max_num_pp = self.sample_loader.max_num_pp
        random.seed(seed)

        samplers.Sampler.__init__(self,
                                  algorithm=getattr(algorithms, task_name),
                                  spec=dfa_specs.DFASPECS[task_name],
                                  num_samples=-1,  #
                                  seed=seed,
                                  if_estimate_max_step=False)

    def _sample_data(self, length: Optional[int] = None, *args, **kwargs):
        sample_id = random.choice(seq=self.sample_id_list)
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

        num_created_samples = 0
        while num_created_samples < num_samples:
            sample_id = self._sample_data(*args, **kwargs)
            print(f'{sample_id} has been sampled... (dfa_sampler)')
            try:
                edge_indices_dict, mask_dict, probes = algorithm(self.sample_loader, sample_id)
            except probing.ProbeError as err:
                if isinstance(err, dfa_utils.YZDExcpetion):
                    print(f'{sample_id} errored!!! error_code: {err.error_code} (dfa_sampler)')
                    continue
                else:
                    print(err)
                    return
            print(f'{sample_id} succeed~~~ (sampler line 92)')
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
        batched_mask_dict = jax.tree_util.tree_map(lambda *x: np.array(x), *mask_dict_list)
        # print('dfa_sampler line 94')    # checked
        # print(f'len of batched_inp_dp_list is: {len(batched_inp_dp_list)}:')
        # for inp_dp in batched_inp_dp_list:
        #     print(f'{inp_dp.name}: {inp_dp.data.shape}')
        # print(f'the shape of batched_trace_o: {batched_trace_o.data.shape}')    # [B, N]
        # print(f'the shape of batched_trace_h: {batched_trace_h.data.shape}')    # [T, B, H]
        # print('in batched_edge_indices_dict:')
        # for key, value in batched_edge_indices_dict.items():
        #     print(f'{key} shape: {value.shape}')    # [B, E, 2]
        # print('in batched_mask_dict:')
        # for key, value in batched_mask_dict.items():
        #     print(f'{key}: {value}')
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
        # print(f'sampler line 110, batch_size = {batch_size}')
        tmp = self._make_batch(
            num_samples=batch_size,
            spec=self._spec,
            min_length=self.expected_hint_len,
            algorithm=self._algorithm)
        # print(f'sampler line 116, the type of tmp is: {type(tmp)}; its len is: {len(tmp)}')
        batched_edge_indices_dict, batched_mask_dict, batched_inp_dp_list, batched_trace_o, batched_trace_h = tmp
        # assert np.array_equal(lengths, sparse_lengths)
        assert len(batched_inp_dp_list) == 6 if self.task_name == 'dfa_liveness' else 5
        print('~~~~~~~~~~ one batch has done! (sampler line 130) ~~~~~~~~~~')
        return Feedback(features=Features(input_dp_list=batched_inp_dp_list,
                                          trace_h=batched_trace_h,
                                          padded_edge_indices_dict=batched_edge_indices_dict,
                                          mask_dict=batched_mask_dict),
                        trace_o=batched_trace_o)


def build_dfa_sampler(task_name: str,
                      sample_id_list: List[str],
                      seed: int,
                      sample_loader: dfa_utils.SampleLoader
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


def FeedbackGenerator(dfa_sampler: DFASampler,
                      batch_size: int):
    while True:
        yield dfa_sampler.next(batch_size=batch_size)


def FeedbackGenerator_limited(dfa_sampler: DFASampler,
                              batch_size: int):
    yield dfa_sampler.next(batch_size=batch_size)
