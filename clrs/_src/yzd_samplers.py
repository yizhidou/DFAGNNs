from absl import logging
from clrs._src import samplers
from clrs._src import probing
from clrs._src import algorithms
from clrs._src import yzd_utils
from clrs._src import yzd_specs
from typing import Dict, Any, Callable, List, Optional, Tuple
from programl.proto import *
import random
import numpy as np

Spec = yzd_specs.Spec
Features = samplers.Features
Feedback = samplers.Feedback


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
        # YZDTODO 在这里应该过滤一下，用max_steps/num_node之类的指标。也许会用一个统计数据去过滤，也许直接用这里的结果进行过滤
        return self.sample_loader, sample_id

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
        inputs, outputs, hints, lengths = self._make_batch(
            num_samples=batch_size,
            spec=self._spec,
            min_length=self.max_steps,
            algorithm=self._algorithm)
        # *args, **kwargs 这两在_make_batch里都是_sample_data的输入，我的_sample_data不需要输入，所以这里就不需要

        # 讲道理在我的情形下这条warning是不应该出现的
        if hints[0].data.shape[0] > self.max_steps:
            logging.warning('Increasing hint lengh from %i to %i',
                            self.max_steps, hints[0].data.shape[0])
            self.max_steps = hints[0].data.shape[0]

        features = Features(inputs=inputs, hints=hints, lengths=lengths)
        return Feedback(features=features, outputs=outputs)


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


def _batch_sparse(sparse_dp_list: samplers.Trajectory,
                  min_steps: int):
    assert sparse_dp_list
    max_steps = min_steps
    batch_size = len(sparse_dp_list)
    dp_name = sparse_dp_list[0].name
    hint_lengths_list = []
    edges_list = []
    nb_nodes_list_with_0 = [0]
    for dp in sparse_dp_list:
        assert dp.name == dp_name
        assert isinstance(dp.data, probing._ArraySparse)
        assert isinstance(dp.data.nb_nodes, int)
        nb_nodes_list_with_0.append(dp.data.nb_nodes)
        assert dp.data.nb_edges.shape[0] == 1
        hint_len = dp.data.nb_edges.shape[1]
        if hint_len > max_steps:
            max_steps = hint_len
        hint_lengths_list.append(hint_len)
        edges_list.append(dp.data.edges)
    cumsum = np.cumsum(nb_nodes_list_with_0)
    edges_batched = np.concatenate(edges_list, axis=0)
    nb_nodes_batched = np.array(nb_nodes_list_with_0[1:])

    nb_edges_batched = np.zeros((batch_size, max_steps))
    for idx, dp in enumerate(sparse_dp_list):
        nb_edges_batched[idx][:dp.data.nb_edges.shape[1]] = dp.data.nb_edges[0]

    indices = np.repeat(np.arange(batch_size), np.sum(nb_edges_batched, axis=1))
    scattered = cumsum[indices]
    edges_batched = edges_batched + np.expand_dims(scattered, axis=1)
    return edges_batched, nb_nodes_batched, nb_edges_batched
