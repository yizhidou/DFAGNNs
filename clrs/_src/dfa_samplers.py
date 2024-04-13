from clrs._src import samplers
from clrs._src import algorithms
from clrs._src import dfa_utils
from clrs._src import specs, dfa_specs
from clrs._src import probing
from typing import List, Optional, Tuple
from programl.proto import *
import os
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
                 # num_samples: int,
                 seed: int,
                 sample_loader: dfa_utils.SampleLoader,
                 iterate_all: bool = False):
        self.sample_id_list = sample_id_list
        self.iterate_all = iterate_all
        self.sample_id_list_iter = None
        self.seed = seed
        self.task_name = task_name
        self.sample_loader = sample_loader
        self.expected_hint_len = self.sample_loader.expected_hint_len
        self.max_num_pp = self.sample_loader.max_num_pp
        # self.sample_id_savepath = sample_id_savepath
        # self.num_sample_id_recorded = 0
        # self.log_sample_id_str = ''
        # if sample_id_savepath is not None and os.path.isfile(sample_id_savepath):
        #     os.system(f'rm {sample_id_savepath}')
        # random.seed(seed)
        # self.sample_id_generator = self.SampleIdGenerator(sample_id_list=sample_id_list,
        #                                                   num_samples=num_samples)
        if self.sample_loader.dfa_version == 0:
            algo = getattr(algorithms, 'dfa')
            spec = dfa_specs.DFASPECS['dfa']
        elif self.sample_loader.dfa_version == 1:
            algo = getattr(algorithms, 'dfa_v1')
            spec = dfa_specs.DFASPECS['dfa_v1']
        elif self.sample_loader.dfa_version == 2:
            algo = getattr(algorithms, 'dfa_v2')
            spec = dfa_specs.DFASPECS['dfa_v2']
            assert self.task_name == 'mix'
        else:
            # assert self.sample_loader.dfa_version is None
            algo = getattr(algorithms, task_name)
            spec = dfa_specs.DFASPECS[self.task_name]
        # algo = getattr(algorithms, 'dfa') if self.sample_loader.if_dfa else getattr(algorithms, task_name)
        samplers.Sampler.__init__(self,
                                  algorithm=algo,
                                  spec=spec,
                                  num_samples=-1,  #
                                  seed=seed,
                                  if_estimate_max_step=False)

    def reset_sample_id_iter(self):
        assert self.iterate_all
        self.sample_id_list_iter = iter(self.sample_id_list)
    def _sample_data(self, length: Optional[int] = None, *args, **kwargs):
        # sample_id = next(self.sample_id_generator)
        # rand_idx = self._rng.randint(0, len(self.sample_id_list) - 1)
        # sampled_id = self.sample_id_list[rand_idx]
        # print(f'dfa_sampler line 63, len of sample_id_list is {len(self.sample_id_list)}; rand_idx = {rand_idx}')
        # print(f'the type of the content in sample_id_list: {type(self.sample_id_list[0])}, and its len: {len(self.sample_id_list[0])}')
        # print(f'the type of sampled_id = {type(sampled_id)}; its len = {len(sampled_id)}')
        if self.iterate_all:
            # print(f'dfa_samplers line 86, len of sample_id_list = {len(self.sample_id_list)}')
            return next(self.sample_id_list_iter)
        return self._rng.choice(self.sample_id_list)

    def _sample_a_batch_sample_ids(self, batch_size=1):
        sampled_ids = []
        if self.iterate_all:
            for _ in range(batch_size):
                sampled_ids.append(next(self.sample_id_list_iter))
            return sampled_ids
        for _ in range(batch_size):
            sampled_ids.append(self._rng.choice(self.sample_id_list))
        return sampled_ids

    def _make_batch(self, num_samples: int,
                    spec: Spec,
                    # min_length: int,
                    algorithm: samplers.Algorithm,
                    if_vali_or_test: bool = False,
                    *args, **kwargs):
        """Generate a batch of data."""
        inp_dp_list_list = []  # pos, if_pp, if_ip, cfg, gen, (kill,) trace_i
        outp_dp_list_list = []  # trace_o
        hint_dp_list_list = []  # trace_h
        edge_indices_dict_list = []
        mask_dict_list = []
        if self.sample_loader.dfa_version == 2:
            if self.sample_loader.balance_sample_task:
                task_idx = self._rng.randint(4)
            else:
                task_idx = self._rng.randint(3)
            if task_idx == 0:
                task_name_for_this_batch = 'liveness'
            elif task_idx == 1:
                task_name_for_this_batch = 'reachability'
            else:
                task_name_for_this_batch = 'dominance'

        num_created_samples = 0
        while num_created_samples < num_samples:
            # if if_vali_or_test:
            #     sample_id = next(self._iter_sample_id_list)
            # else:
            #     sample_id = self._sample_data(*args, **kwargs)
            sample_id = self._sample_data(*args, **kwargs)
            print(f'{sample_id} has been sampled...(dfa_samplers line 110)')
            # self.num_sample_id_recorded += 1
            # self.log_sample_id_str += f'{sample_id}\n'
            # if self.num_sample_id_recorded % 500 == 0:
            #     with open(self.sample_id_savepath, 'a') as sample_id_logger:
            #         sample_id_logger.write(self.log_sample_id_str)
            #     self.log_sample_id_str = ''
            # print(f'dfa_sampler line 109, dfa_version = {self.sample_loader.dfa_version}')
            try:
                if self.sample_loader.dfa_version is not None:
                    edge_indices_dict, mask_dict, probes = algorithm(self.sample_loader, sample_id,
                                                                     task_name_for_this_batch if self.sample_loader.dfa_version == 2 else self.task_name)
                else:
                    edge_indices_dict, mask_dict, probes = algorithm(self.sample_loader, sample_id)
            except probing.ProbeError as err:
                if isinstance(err, dfa_utils.DFAException):
                    print(f'{sample_id} errored!!! error_code: {err.error_code} (dfa_sampler line 98)')
                    continue
                else:
                    print(err)
                    return
            # print(f'{sample_id} succeed~~~ (dfa_sampler line 92)')
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
        if self.sample_loader.dfa_version == 2:
            return task_name_for_this_batch, batched_edge_indices_dict, batched_mask_dict, batched_inp_dp_list, batched_trace_o, batched_trace_h
        else:
            return None, batched_edge_indices_dict, batched_mask_dict, batched_inp_dp_list, batched_trace_o, batched_trace_h

    def next(self, batch_size: Optional[int] = None,
             # if_vali_or_test: bool = False
             ) -> Feedback:
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
            # min_length=self.expected_hint_len,
            algorithm=self._algorithm,
            # if_vali_or_test=if_vali_or_test
        )
        # print(f'sampler line 116, the type of tmp is: {type(tmp)}; its len is: {len(tmp)}')
        task_name_for_this_batch, batched_edge_indices_dict, batched_mask_dict, batched_inp_dp_list, batched_trace_o, batched_trace_h = tmp
        return task_name_for_this_batch, Feedback(features=Features(input_dp_list=batched_inp_dp_list,
                                                                        trace_h=batched_trace_h,
                                                                        padded_edge_indices_dict=batched_edge_indices_dict,
                                                                        mask_dict=batched_mask_dict),
                                                      trace_o=batched_trace_o)
    def _make_batch_all_tasks(self, num_samples: int,
                    spec: Spec,
                    algorithm: samplers.Algorithm,
                    *args, **kwargs):
        liveness_result, reachability_result, dominance_result = None, None, None
        task_names_list = ['liveness', 'reachability', 'dominance']
        inp_dp_list_list = [[], [], []]  # pos, if_pp, if_ip, cfg, gen, (kill,) trace_i
        outp_dp_list_list = [[], [], []]  # trace_o
        hint_dp_list_list = [[], [], []]  # trace_h
        edge_indices_dict_list = [[], [], []]
        mask_dict_list = [[], [], []]
        sample_ids_this_batch = []
        num_created_samples = 0
        while num_created_samples < num_samples:
            sample_id = self._sample_data(*args, **kwargs)
            print(f'{sample_id} has been sampled...(dfa_samplers line 110)')
            for task_idx, task_name_for_this_batch in enumerate(task_names_list):
                try:
                    print(f'{task_name_for_this_batch} is on going...')
                    edge_indices_dict, mask_dict, probes = algorithm(self.sample_loader, sample_id, task_name_for_this_batch)
                except probing.ProbeError as err:
                    if isinstance(err, dfa_utils.DFAException):
                        print(f'{sample_id} errored!!! for task {task_name_for_this_batch}. error_code: {err.error_code} (dfa_sampler line 98)')
                        continue
                    else:
                        print(err)
                        return
                edge_indices_dict_list[task_idx].append(edge_indices_dict)
                mask_dict_list[task_idx].append(mask_dict)
                inp_dp_list, outp_dp_list, hint_dp_list = probing.split_stages(probes, spec)
                inp_dp_list_list[task_idx].append(inp_dp_list)
                outp_dp_list_list[task_idx].append(outp_dp_list)
                hint_dp_list_list[task_idx].append(hint_dp_list)
            # print(f'{sample_id} succeed~~~ (dfa_sampler line 92)')
            num_created_samples += 1
            sample_ids_this_batch.append(sample_id)
        for task_idx, task_name in enumerate(task_names_list):
            # if len(inp_dp_list_list[task_idx]) == 0:
            #     assert len(outp_dp_list_list[task_idx]) == 0
            #     assert len(hint_dp_list_list[task_idx]) == 0
            #     assert len(hint_dp_list_list[task_idx]) == 0
            #     continue
            batched_inp_dp_list = _batch_ioh(inp_dp_list_list[task_idx])
            batched_trace_o = _batch_ioh(outp_dp_list_list[task_idx])[0]
            batched_trace_h = _batch_ioh(hint_dp_list_list[task_idx])[0]
            batched_edge_indices_dict = jax.tree_util.tree_map(lambda *x: np.stack(x), *edge_indices_dict_list[task_idx])
            batched_mask_dict = jax.tree_util.tree_map(lambda *x: np.array(x), *mask_dict_list[task_idx])
            result_of_this_task = (task_name, batched_edge_indices_dict, batched_mask_dict, batched_inp_dp_list, batched_trace_o, batched_trace_h)
            if task_name == 'liveness':
                liveness_result = result_of_this_task
            elif task_name == 'reachability':
                reachability_result = result_of_this_task
            else:
                dominance_result = result_of_this_task
        return sample_ids_this_batch, (liveness_result, reachability_result, dominance_result)

    def next_all_tasks(self, batch_size: Optional[int] = None):
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
        sample_ids_this_batch, results = self._make_batch_all_tasks(
            num_samples=batch_size,
            spec=self._spec,
            # min_length=self.expected_hint_len,
            algorithm=self._algorithm,
            # if_vali_or_test=if_vali_or_test
        )
        # print(f'sampler line 116, the type of tmp is: {type(tmp)}; its len is: {len(tmp)}')
        liveness_batch, reachability_batch, dominance_batch = None, None, None
        for result in results:
            if result is None:
                continue
            task_name_for_this_batch, batched_edge_indices_dict, batched_mask_dict, batched_inp_dp_list, batched_trace_o, batched_trace_h = result
            batch = (sample_ids_this_batch, task_name_for_this_batch, Feedback(features=Features(input_dp_list=batched_inp_dp_list,
                                                                        trace_h=batched_trace_h,
                                                                        padded_edge_indices_dict=batched_edge_indices_dict,
                                                                        mask_dict=batched_mask_dict),
                                                      trace_o=batched_trace_o))
            if task_name_for_this_batch == 'liveness':
                liveness_batch = batch
            elif task_name_for_this_batch == 'reachability':
                reachability_batch = batch
            else:
                dominance_batch = batch

        return liveness_batch, reachability_batch, dominance_batch
def _batch_ioh(ioh_dp_list_list: Trajectories) -> Trajectory:
    assert ioh_dp_list_list
    for sample_idx, dp_list_one_sample in enumerate(ioh_dp_list_list):
        for dp_idx, dp in enumerate(dp_list_one_sample):
            # print(f'dfa_sampler line 153, {dp.name}: {dp.data.shape}')
            if dp.name == 'trace_h':
                assert dp.data.shape[1] == 1
                concat_dim = 1
            else:
                assert dp.data.shape[0] == 1
                concat_dim = 0
    return jax.tree_util.tree_map(lambda *x: np.concatenate(x,
                                                            axis=concat_dim),
                                  *ioh_dp_list_list)


# def build_dfa_sampler(task_name: str,
#                       sample_id_list: List[str],
#                       seed: int,
#                       sample_loader: dfa_utils.SampleLoader
#                       ) -> Tuple[DFASampler, Spec]:
#     """Builds a sampler. See `Sampler` documentation."""
#
#     assert task_name in ['liveness', 'dominance', 'reachability']
#     print(f'dfa_sampler line 171 if_dfa = {sample_loader.if_dfa}')
#     if sample_loader.if_dfa:
#         spec = dfa_specs.DFASPECS['dfa']
#     else:
#         spec = dfa_specs.DFASPECS[task_name]
#
#     sampler = DFASampler(task_name=task_name,
#                          sample_id_list=sample_id_list,
#                          seed=seed,
#                          sample_loader=sample_loader)
#     return sampler, spec


def FeedbackGenerator(dfa_sampler: DFASampler,
                      batch_size: int,
                      # if_vali_or_test: bool = False
                      ):
    while True:
        yield dfa_sampler.next(batch_size=batch_size)

def FeedbackGenerator_all_tasks(dfa_sampler: DFASampler,
                              batch_size: int):
    while True:
        liveness_batch, reachability_batch, dominance_batch = dfa_sampler.next_all_tasks(batch_size=batch_size)
        yield liveness_batch
        yield reachability_batch
        yield dominance_batch
