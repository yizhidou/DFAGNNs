from absl import logging
from clrs._src import samplers
from clrs._src import probing
from clrs._src import algorithms
from clrs._src import yzd_utils
from clrs._src import specs, yzd_specs
from clrs._src import yzd_probing
from typing import Dict, Any, Callable, List, Optional, Tuple, Iterable
from programl.proto import *
import random
import numpy as np
import jax
import collections

Spec = yzd_specs.Spec
# Features = collections.namedtuple('Features', ['dense_inputs',
#                                                'sparse_inputs',
#                                                'dense_hints',
#                                                'sparse_hints',
#                                                'lengths'])
Features = collections.namedtuple('Features', ['input_NODE_dp_list',
                                               'input_EDGE_dp_list',
                                               'trace_h',
                                               'hint_len'])

Feedback = collections.namedtuple('Feedback', ['features', 'trace_o'])
# DenseFeatures = samplers.Features
# SparseFeatures = collections.namedtuple('SparseFeatures', ['sparse_inputs', 'sparse_hints', 'sparse_lengths'])
# YZDFeedback = collections.namedtuple('YZDFeedback',
#                                      ['dense_features',
#                                       # 'dense_outputs',
#                                       'sparse_features',
#                                       'sparse_outputs'])

_ArraySparse = yzd_probing.ArraySparse
_ArrayDense = yzd_probing.ArrayDense
_Array = yzd_probing.Array
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
        self.max_steps = self.sample_loader.max_iteration - 1
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
        input_NODE_dp_list_list = []
        input_EDGE_dp_list_list = []
        trace_o_list_list = []
        trace_h_list = []

        num_created_samples = 0
        while num_created_samples < num_samples:
            sample_id = self._sample_data(*args, **kwargs)
            try:
                _, probes = algorithm(self.sample_loader, sample_id)
            except yzd_utils.YZDExcpetion:
                continue
            num_created_samples += 1
            inp_NODE_dp_list, inp_EDGE_dp_list, trace_o_list, trace_h = yzd_probing.yzd_split_stages(probes, spec)
            input_NODE_dp_list_list.append(inp_NODE_dp_list)
            # outputs.append(outp)  # this should be empty
            # hints.append(hint)
            input_EDGE_dp_list_list.append(inp_EDGE_dp_list)
            trace_o_list_list.append(trace_o_list)
            trace_h_list.append(trace_h)

        # Batch and pad trajectories to max(T).
        batched_input_NODE_dp_list = _batch_NODE_input(input_NODE_dp_list_list)
        batched_input_EDGE_dp_list = _batch_EDGE_io(input_EDGE_dp_list_list)
        batched_trace_o = _batch_EDGE_io(trace_o_list_list)[0]
        batched_trace_h, hint_len = _batch_trace_h(trace_h_traj=trace_h_list,
                                                   batched_trace_o=batched_trace_o,
                                                   min_steps=min_length)
        # outputs = samplers._batch_io(outputs)
        # hints, lengths = samplers._batch_hints(hints, min_length)
        # sparse_inputs = _batch_io_sparse(sparse_inputs)
        # sparse_outputs = _batch_io_sparse(sparse_outputs)
        # sparse_hints, sparse_lengths = _batch_hints_sparse(sparse_hints, min_length)
        return batched_input_NODE_dp_list, batched_input_EDGE_dp_list, batched_trace_o, batched_trace_h, hint_len

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
        batched_input_NODE_dp_list, batched_input_EDGE_dp_list, batched_trace_o, batched_trace_h, hint_len = self._make_batch(
            num_samples=batch_size,
            spec=self._spec,
            min_length=self.max_steps,
            algorithm=self._algorithm)
        # assert np.array_equal(lengths, sparse_lengths)
        assert len(batched_input_NODE_dp_list) == 3 and len(
            batched_input_EDGE_dp_list) == 3 if self.task_name == 'yzd_liveness' else 2

        # 讲道理在我的情形下这条warning是不应该出现的
        # if hints[0].data.shape[0] > self.max_steps:
        #     logging.warning('Increasing hint lengh from %i to %i',
        #                     self.max_steps, hints[0].data.shape[0])
        #     self.max_steps = hints[0].data.shape[0]

        return Feedback(features=Features(input_NODE_dp_list=batched_input_NODE_dp_list,
                                          input_EDGE_dp_list=batched_input_EDGE_dp_list,
                                          trace_h=batched_trace_h,
                                          hint_len=hint_len),
                        trace_o=batched_trace_o)


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


def _batch_NODE_input(input_NODE_dp_list_list: Trajectories):
    assert input_NODE_dp_list_list
    for input_NODE_dp_list in input_NODE_dp_list_list:
        for i, input_NODE_dp in enumerate(input_NODE_dp_list):
            assert input_NODE_dp.name == input_NODE_dp_list_list[0][i].name
    return jax.tree_util.tree_map(lambda *x: np.concatenate(x), *input_NODE_dp_list_list)


def _batch_EDGE_io(io_EDGE_dp_list_list: Trajectories):
    assert io_EDGE_dp_list_list
    for input_EDGE_dp_list in io_EDGE_dp_list_list:
        for i, input_EDGE_dp in enumerate(input_EDGE_dp_list):
            # assert isinstance(dp.data, _ArraySparse)
            assert io_EDGE_dp_list_list[0][i].name == input_EDGE_dp.name
    return jax.tree_util.tree_map(lambda *x: _batch_EDGE_one_probe(x),
                                  *io_EDGE_dp_list_list)


def _batch_EDGE_one_probe(EDGE_data_list):
    '''
    EDGE_dp_list: Iterable[_ArraySparse]
    '''
    assert EDGE_data_list
    batch_size = len(EDGE_data_list)
    edges_list = []
    nb_nodes_list_with_0 = [0]
    nb_edges_list = []
    for idx, dp_data in enumerate(EDGE_data_list):
        nb_nodes_list_with_0.append(dp_data.nb_nodes)
        nb_edges_list.append(dp_data.nb_edges)
        edges_list.append(dp_data.edge_indices_with_optional_content)
    cumsum = np.cumsum(nb_nodes_list_with_0)
    edges_batched = np.concatenate(edges_list, axis=0)
    nb_nodes_batched = np.array(nb_nodes_list_with_0[1:])
    nb_edges_batched = np.array(nb_edges_list)
    indices = np.repeat(np.arange(batch_size), nb_edges_batched)
    scattered = cumsum[indices]
    edges_batched[:, :2] += np.expand_dims(scattered, axis=1)
    return _ArraySparse(edges_with_optional_content=edges_batched,
                        nb_nodes=nb_nodes_batched,
                        nb_edges=nb_edges_batched)


def _batch_trace_h(trace_h_traj: Trajectory,
                   batched_trace_o: _DataPoint,
                   min_steps: int):
    # batch trace_h_sparce
    assert trace_h_traj
    batch_size = len(trace_h_traj)
    padded_trace_h_edges = np.repeat(a=np.expand_dims(batched_trace_o.data.edges_with_optional_content,
                                                      axis=0),
                                     repeats=min_steps,
                                     axis=0)  # [min_steps, nb_edges_entire_batch, 3]

    start_nb_edges = 0
    accum_nb_nodes = 0
    hint_len = np.zeros(batch_size, dtype=int)
    for dp_idx, dp in enumerate(trace_h_traj):
        assert dp.name == 'trace_h_sparse'
        dp_hint_len, dp_nb_edges, = dp.data.edges_with_optional_content.shape[:2]
        assert dp_nb_edges == dp.data.nb_edges
        tmp = accum_nb_nodes * np.ones_like(dp.data.edges_with_optional_content)
        tmp[..., -1] = 0
        padded_trace_h_edges[:dp_hint_len, start_nb_edges:start_nb_edges + dp_nb_edges,
        :] = dp.data.edges_with_optional_content + tmp
        start_nb_edges += dp_nb_edges
        assert batched_trace_o.data.nb_nodes[dp_idx] == dp.data.nb_nodes
        assert batched_trace_o.data.nb_edges[dp_idx] == dp_nb_edges
        accum_nb_nodes += dp.data.nb_nodes
        hint_len[dp_idx] = dp_hint_len

    padded_trace_h = _ArraySparse(edges_with_optional_content=padded_trace_h_edges,
                                  nb_nodes=batched_trace_o.data.nb_nodes,
                                  nb_edges=batched_trace_o.data.nb_edges)
    return _DataPoint(name='trace_h_sparse',
                      location=specs.Location.EDGE,
                      type_=specs.Type.MASK,
                      data=padded_trace_h), \
           hint_len
