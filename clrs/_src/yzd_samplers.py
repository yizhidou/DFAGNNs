from absl import logging
from clrs._src import samplers
from clrs._src import probing
from clrs._src import algorithms
from clrs._src import yzd_utils
from clrs._src import yzd_specs
from clrs._src import yzd_probing
from typing import Dict, Any, Callable, List, Optional, Tuple
from programl.proto import *
import random
import numpy as np
import collections

Spec = yzd_specs.Spec
Features = samplers.Features
Feedback = samplers.Feedback
# DenseFeatures = samplers.Features
# SparseFeatures = collections.namedtuple('SparseFeatures', ['sparse_inputs', 'sparse_hints', 'sparse_lengths'])
# YZDFeedback = collections.namedtuple('YZDFeedback',
#                                      ['dense_features',
#                                       # 'dense_outputs',
#                                       'sparse_features',
#                                       'sparse_outputs'])

_ArraySparse = yzd_probing._ArraySparse
_ArrayDense = yzd_probing._ArrayDense
_Array = yzd_probing._Array
_DataPoint = yzd_probing.DataPoint
Trajectory = List[_DataPoint]
Trajectories = List[Trajectory]


class YZDSampler(samplers.Sampler):

    # 要设定max_step的
    def __init__(self,
                 task_name: str,
                 sample_id_list: List[str],
                 seed: int,
                 sample_loader: yzd_utils.SampleLoader
                 ):
        if not task_name in ['yzd_liveness', 'yzd_dominance', 'yzd_reachability']:
            raise NotImplementedError(f'No implementation of algorithm {task_name}.')
        self.sample_id_list = sample_id_list
        self.task_name = task_name
        self.sample_loader = sample_loader
        self.max_steps = self.sample_loader.max_iteration
        self.max_num_pp = self.sample_loader.max_num_pp

        samplers.Sampler.__init__(self,
                                  algorithm=getattr(algorithms, task_name),
                                  spec=yzd_specs.YZDSPECS[task_name],
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
        inputs = []
        # outputs = []
        hints = []
        sparse_inputs = []
        sparse_outputs = []
        sparse_hints = []

        num_created_samples = 0
        while num_created_samples < num_samples:
            sample_id = self._sample_data(*args, **kwargs)
            try:
                _, probes = algorithm(self.sample_loader, sample_id)
            except yzd_utils.YZDExcpetion:
                continue
            num_created_samples += 1
            inp, hint, s_inp, s_outp, s_hint = yzd_probing.yzd_split_stages(probes, spec)
            inputs.append(inp)
            # outputs.append(outp)  # this should be empty
            hints.append(hint)
            sparse_inputs.append(s_inp)
            sparse_outputs.append(s_outp)
            sparse_hints.append(s_hint)
            if len(hints) % 1000 == 0:
                logging.info('%i samples created', len(hints))

        # Batch and pad trajectories to max(T).
        inputs = samplers._batch_io(inputs)
        # outputs = samplers._batch_io(outputs)
        hints, lengths = samplers._batch_hints(hints, min_length)

        sparse_inputs = _batch_io_sparse(sparse_inputs)
        sparse_outputs = _batch_io_sparse(sparse_outputs)
        sparse_hints, sparse_lengths = _batch_hints_sparse(sparse_hints, min_length)
        return inputs, hints, lengths, sparse_inputs, sparse_outputs, sparse_hints, sparse_lengths

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
        inputs, hints, lengths, sparse_inputs, sparse_outputs, sparse_hints, sparse_lengths = self._make_batch(
            num_samples=batch_size,
            spec=self._spec,
            min_length=self.max_steps,
            algorithm=self._algorithm)
        assert np.array_equal(lengths, sparse_lengths)
        if self.task_name == 'yzd_liveness':
            assert len(inputs) == 3 and len(hints) == 1 and len(sparse_inputs) == 3 and len(sparse_hints) == 1
        else:
            assert len(inputs) == 3 and len(hints) == 1 and len(sparse_inputs) == 2 and len(sparse_hints) == 1

        # 讲道理在我的情形下这条warning是不应该出现的
        if hints[0].data.shape[0] > self.max_steps:
            logging.warning('Increasing hint lengh from %i to %i',
                            self.max_steps, hints[0].data.shape[0])
            self.max_steps = hints[0].data.shape[0]

        # dense_features = DenseFeatures(inputs=inputs, hints=hints, lengths=lengths)
        # sparse_featrures = SparseFeatures(sparse_inputs=sparse_inputs, sparse_hints=sparse_hints,
        #                                   sparse_lengths=sparse_lengths)
        # return YZDFeedback(dense_features=dense_features,
        #                    # dense_outputs=outputs,
        #                    sparse_features=sparse_featrures, sparse_outputs=sparse_outputs)
        return Feedback(features=Features(inputs=inputs + sparse_inputs,
                                          hints=hints,
                                          lengths=lengths),
                        outputs=sparse_outputs)


def build_yzd_sampler(task_name: str,
                      sample_id_list: List[str],
                      seed: int,
                      sample_loader: yzd_utils.SampleLoader
                      ) -> Tuple[YZDSampler, Spec]:
    """Builds a sampler. See `Sampler` documentation."""

    if task_name not in ['yzd_liveness', 'yzd_dominance', 'yzd_reachability']:
        raise NotImplementedError(f'No implementation of algorithm {task_name}.')
    spec = yzd_specs.YZDSPECS[task_name]

    sampler = YZDSampler(task_name=task_name,
                         sample_id_list=sample_id_list,
                         seed=seed,
                         sample_loader=sample_loader,
                         )
    return sampler, spec


def _batch_io_sparse(sparse_io_traj_list: Trajectories):
    assert sparse_io_traj_list
    batched_sparse_io_list = []
    for sparse_io_traj in sparse_io_traj_list:
        batched_sparse_io_list.append(_batch_io_sparse_one_probe(sparse_io_traj))
    return batched_sparse_io_list


def _batch_io_sparse_one_probe(sparse_io_traj: Trajectory):
    assert sparse_io_traj
    dp_name = sparse_io_traj[0].name
    batch_size = len(sparse_io_traj)
    edges_list = []
    nb_nodes_list_with_0 = [0]
    nb_edges_list = []
    for idx, dp in enumerate(sparse_io_traj):
        assert isinstance(dp, _ArraySparse)
        assert dp.name == dp_name
        assert isinstance(dp.data.nb_nodes, int)
        nb_nodes_list_with_0.append(dp.data.nb_nodes)
        assert dp.data.nb_edges.shape[0] == 1 and dp.data.nb_edges.shape[1] == 1
        nb_edges_list.append(dp.data.nb_edges)
        edges_list.append(dp.data.edge_indices_with_optional_content)
    cumsum = np.cumsum(nb_nodes_list_with_0)
    edges_batched = np.concatenate(edges_list, axis=0)
    nb_nodes_batched = np.array(nb_nodes_list_with_0[1:])
    nb_edges_batched = np.concatenate(nb_edges_list)
    indices = np.repeat(np.arange(batch_size), np.sum(nb_edges_batched, axis=1))
    scattered = cumsum[indices]
    edges_batched = edges_batched + np.expand_dims(scattered, axis=1)
    return _ArraySparse(edge_indices_with_optional_content=edges_batched,
                        nb_nodes=nb_nodes_batched,
                        nb_edges=nb_edges_batched)


def _batch_hints_sparse(sparse_hint_traj_list: Trajectories, min_steps: int):
    assert sparse_hint_traj_list
    batched_sparse_hint_list = []
    hint_lengths = []
    for sparse_hints_traj in sparse_hint_traj_list:
        batched_sparse_hint_one_probe, hint_len = _batch_hints_sparse_one_probe(sparse_hint_traj=sparse_hints_traj,
                                                                                min_steps=min_steps)
        hint_lengths.append(hint_len)
    return batched_sparse_hint_list, np.array(hint_lengths)


def _batch_hints_sparse_one_probe(sparse_hint_traj: Trajectory,
                                  min_steps: int):
    assert sparse_hint_traj
    max_steps = min_steps
    batch_size = len(sparse_hint_traj)
    dp_name = sparse_hint_traj[0].name
    hint_len_this_probe = sparse_hint_traj[0].data.nb_edges.shape[1]
    edges_list = []
    nb_nodes_list_with_0 = [0]
    for dp in sparse_hint_traj:
        assert dp.name == dp_name
        assert isinstance(dp.data, _ArraySparse)
        assert isinstance(dp.data.nb_nodes, int)
        nb_nodes_list_with_0.append(dp.data.nb_nodes)
        assert dp.data.nb_edges.shape[0] == 1 and dp.data.nb_edges.shape[1] == hint_len_this_probe
        # hint_len = dp.data.nb_edges.shape[1]
        # if hint_len > max_steps:
        #     max_steps = hint_len
        # hint_lengths_list.append(hint_len)
        edges_list.append(dp.data.edge_indices_with_optional_content)
    cumsum = np.cumsum(nb_nodes_list_with_0)
    edges_batched = np.concatenate(edges_list, axis=0)
    nb_nodes_batched = np.array(nb_nodes_list_with_0[1:])

    nb_edges_batched = np.zeros((batch_size, max_steps))
    for idx, dp in enumerate(sparse_hint_traj):
        nb_edges_batched[idx][:dp.data.nb_edges.shape[1]] = dp.data.nb_edges[0]

    indices = np.repeat(np.arange(batch_size), np.sum(nb_edges_batched, axis=1))
    scattered = cumsum[indices]
    edges_batched = edges_batched + np.expand_dims(scattered, axis=1)
    return _ArraySparse(edge_indices_with_optional_content=edges_batched,
                        nb_nodes=nb_nodes_batched,
                        nb_edges=nb_edges_batched), \
           hint_len_this_probe
