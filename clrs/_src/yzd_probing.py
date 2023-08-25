from clrs._src import probing
from clrs._src import specs
import collections
import numpy as np
from typing import Dict, List, Tuple, Union
import jax
import attr

_Location = specs.Location
_Stage = specs.Stage
_Type = specs.Type
ArraySparse = collections.namedtuple('ArraySparse', ['edge_indices_with_optional_content',
                                                     'nb_nodes',
                                                     'nb_edges'])
ArrayDense = np.ndarray
Array = Union[ArrayDense, ArraySparse]
_Data = Union[Array, List[Array]]
_DataOrType = Union[_Data, str]

ProbesDict = Dict[
    str, Dict[str, Dict[str, Dict[str, _DataOrType]]]]

ProbeError = probing.ProbeError


# First anotation makes this object jax.jit/pmap friendly, second one makes this
# tf.data.Datasets friendly.
@jax.tree_util.register_pytree_node_class
@attr.define
class DataPoint:
    """Describes a data point."""

    _name: str
    _location: str
    _type_: str
    data: Array

    @property
    def name(self):
        return probing._convert_to_str(self._name)

    @property
    def location(self):
        return probing._convert_to_str(self._location)

    @property
    def type_(self):
        return probing._convert_to_str(self._type_)

    def __repr__(self):
        s = f'DataPoint(name="{self.name}",\tlocation={self.location},\t'
        if isinstance(self.data, ArraySparse):
            s += f'data=ArrarySparse(nb_nodes={sum(self.data.nb_nodes)}))\t'
        else:
            assert isinstance(self.data, ArrayDense)
            s += f'data=ArrayDense({self.data.shape}))'
        return s

    def tree_flatten(self):
        if isinstance(self.data, ArrayDense):
            data = (self.data,)
            meta = (self.name, self.location, self.type_)
        else:
            data = (self.data.edge_indices_with_optional_content,)
            meta = (self.data.nb_nodes, self.data.nb_edges, self.name, self.location, self.type_)
        return data, meta

    @classmethod
    def tree_unflatten(cls, meta, data):
        if isinstance(data, ArrayDense):
            name, location, type_ = meta
            subdata, = data
        else:
            nb_nodes, nb_edges, name, location, type_ = meta
            subdata = ArraySparse(edge_indices_with_optional_content=data,
                                  nb_nodes=nb_nodes,
                                  nb_edges=nb_edges)
        return DataPoint(name, location, type_, subdata)


def yzd_push(probes: ProbesDict, stage: str, next_probe):
    """Pushes a probe into an existing `ProbesDict`."""
    for loc in [_Location.NODE, _Location.EDGE, _Location.GRAPH]:
        for name in probes[stage][loc]:
            if name not in next_probe:
                raise ProbeError(f'Missing probe for {name}.')
            if isinstance(probes[stage][loc][name]['data'], ArraySparse) or isinstance(
                    probes[stage][loc][name]['data'], ArrayDense):
                raise ProbeError('Attemping to push to finalized `ProbesDict`.')
            # Pytype thinks initialize() returns a ProbesDict with a str for all final
            # values instead of _DataOrType.
            probes[stage][loc][name]['data'].append(next_probe[name])  # pytype: disable=attribute-error


def yzd_finalize_deprecated(probes: ProbesDict):
    """Finalizes a `ProbesDict` by stacking/squeezing `data` field."""
    # 把ProbesDict给整理一下: 属于HINT的probe给stack成一个dense array或者合并成一个sparse array;
    # 不属于HINT的probe每个单独成为一个dense array或者sparse array
    for stage in [_Stage.INPUT, _Stage.OUTPUT, _Stage.HINT]:
        for loc in [_Location.NODE, _Location.EDGE, _Location.GRAPH]:
            for name in probes[stage][loc]:
                if isinstance(probes[stage][loc][name]['data'], ArraySparse) or isinstance(
                        probes[stage][loc][name]['data'], ArrayDense):
                    raise ProbeError('Attemping to re-finalize a finalized `ProbesDict`.')
                if stage == _Stage.HINT:
                    # Hints are provided for each timestep. Stack them here.
                    if loc == _Location.EDGE:
                        edge_indices_with_optional_content_list = []
                        nb_edges_list = []
                        nb_nodes = probes[stage][loc][name]['data'][0].nb_nodes
                        for hint_idx, dp in enumerate(probes[stage][loc][name]['data']):
                            assert isinstance(dp, ArraySparse)
                            assert dp.edge_indices_with_optional_content.ndim == 2
                            assert dp.edge_indices_with_optional_content.shape[-1] == 3
                            edge_indices_with_optional_content_list.append(dp.edge_indices_with_optional_content)
                            assert isinstance(dp.nb_edges, int)
                            nb_edges_list.append(dp.nb_edges)
                            assert dp.nb_nodes == nb_nodes
                        probes[stage][loc][name]['data'] = ArraySparse(
                            edge_indices_with_optional_content=np.concatenate(edge_indices_with_optional_content_list,
                                                                              axis=0),
                            # [nb_edges_total, 2 or 3]
                            nb_nodes=nb_nodes,
                            nb_edges=np.expand_dims(np.array(nb_edges_list),
                                                    0)  # [1, hint_len]
                        )

                    else:
                        assert name == 'time'
                        probes[stage][loc][name]['data'] = np.stack(
                            probes[stage][loc][name]['data'])  # [hint_len, ]
                else:
                    # Only one instance of input/output exist. Remove leading axis.
                    assert len(probes[stage][loc][name]['data']) == 1
                    if loc == _Location.EDGE:
                        assert isinstance(probes[stage][loc][name]['data'][0], ArraySparse)

                        probes[stage][loc][name]['data'] = ArraySparse(
                            edge_indices_with_optional_content=probes[stage][loc][name][
                                'data'][0].edge_indices_with_optional_content,
                            nb_nodes=probes[stage][loc][name][
                                'data'][0].nb_nodes,
                            nb_edges=np.array([[probes[stage][loc][name][
                                                    'data'][0].nb_edges]])
                            # [1, 1]
                        )
                    else:
                        assert isinstance(probes[stage][loc][name]['data'][0], ArrayDense)
                        probes[stage][loc][name]['data'] = np.squeeze(
                            np.array(probes[stage][loc][name]['data']))


def yzd_finalize(probes: ProbesDict):
    """Finalizes a `ProbesDict` by stacking/squeezing `data` field."""
    # 把ProbesDict给整理一下: 属于HINT的probe给stack成一个dense array或者合并成一个sparse array;
    # 不属于HINT的probe每个单独成为一个dense array或者sparse array
    for stage in [_Stage.INPUT, _Stage.OUTPUT, _Stage.HINT]:
        for loc in [_Location.NODE, _Location.EDGE, _Location.GRAPH]:
            for name in probes[stage][loc]:
                if isinstance(probes[stage][loc][name]['data'], ArraySparse) or isinstance(
                        probes[stage][loc][name]['data'], ArrayDense):
                    raise ProbeError('Attemping to re-finalize a finalized `ProbesDict`.')
                if stage == _Stage.HINT:
                    # Hints are provided for each timestep. Stack them here.
                    if loc == _Location.EDGE:
                        assert name == 'trace_i_sparce'
                        edge_indices_with_optional_content_list = []
                        nb_edges = probes[stage][loc][name]['data'][0].nb_edges
                        nb_nodes = probes[stage][loc][name]['data'][0].nb_nodes
                        for hint_idx, dp in enumerate(probes[stage][loc][name]['data']):
                            assert isinstance(dp, ArraySparse)
                            assert dp.edge_indices_with_optional_content.ndim == 2

                            edge_indices_with_optional_content_list.append(dp.edge_indices_with_optional_content)
                            assert dp.nb_edges == nb_edges
                            assert dp.nb_nodes == nb_nodes
                        probes[stage][loc][name]['data'] = ArraySparse(
                            edge_indices_with_optional_content=np.stack(edge_indices_with_optional_content_list),
                            # [hint_len, nb_edges(num_pp*num_ip), 3]
                            nb_nodes=nb_nodes,
                            nb_edges=nb_edges
                        )

                    else:
                        assert name == 'time'
                        probes[stage][loc][name]['data'] = np.stack(
                            probes[stage][loc][name]['data'])  # [hint_len, ]
                else:
                    # Only one instance of input/output exist. Remove leading axis.
                    assert len(probes[stage][loc][name]['data']) == 1
                    if loc == _Location.EDGE:
                        assert name in ['cfg_sparse', 'gen_sparse', 'kill_sparse',
                                        'trace_i_sparse', 'trace_o_sparse']
                        assert isinstance(probes[stage][loc][name]['data'][0], ArraySparse)

                        probes[stage][loc][name]['data'] = ArraySparse(
                            edge_indices_with_optional_content=probes[stage][loc][name][
                                'data'][0].edge_indices_with_optional_content,
                            # [nb_edges(nb_cfg_edges/num_pp*num_ip), 2/3]
                            nb_nodes=probes[stage][loc][name][
                                'data'][0].nb_nodes,
                            nb_edges=probes[stage][loc][name]['data'][0].nb_edges)
                    else:
                        assert isinstance(probes[stage][loc][name]['data'][0], ArrayDense)
                        probes[stage][loc][name]['data'] = np.squeeze(
                            np.array(probes[stage][loc][name]['data']))


def yzd_split_stages(probes: ProbesDict,
                     spec: specs.Spec):
    """Splits contents of `ProbesDict` into `DataPoint`s by stage."""

    inputs = []
    # outputs = []
    hints = []

    sparse_inputs = []
    sparse_outputs = []
    sparse_hints = []

    for name in spec:
        stage, loc, t = spec[name]

        if stage not in probes:
            raise ProbeError(f'Missing stage {stage}.')
        if loc not in probes[stage]:
            raise ProbeError(f'Missing location {loc}.')
        if name not in probes[stage][loc]:
            raise ProbeError(f'Missing probe {name}.')
        if 'type_' not in probes[stage][loc][name]:
            raise ProbeError(f'Probe {name} missing attribute `type_`.')
        if 'data' not in probes[stage][loc][name]:
            raise ProbeError(f'Probe {name} missing attribute `data`.')
        if t != probes[stage][loc][name]['type_']:
            raise ProbeError(f'Probe {name} of incorrect type {t}.')

        data = probes[stage][loc][name]['data']
        if not isinstance(probes[stage][loc][name]['data'], ArraySparse) and not isinstance(
                probes[stage][loc][name]['data'], ArrayDense):
            raise ProbeError((f'Invalid `data` for probe "{name}". ' +
                              'Did you forget to call `probing.finalize`?'))

        if t in [_Type.MASK, _Type.MASK_ONE, _Type.CATEGORICAL]:
            # pytype: disable=attribute-error
            if not ((data == 0) | (data == 1) | (data == -1)).all():
                raise ProbeError(f'0|1|-1 `data` for probe "{name}"')
            # pytype: enable=attribute-error
            if t in [_Type.MASK_ONE, _Type.CATEGORICAL
                     ] and not np.all(np.sum(np.abs(data), -1) == 1):
                raise ProbeError(f'Expected one-hot `data` for probe "{name}"')

        if loc == _Location.EDGE:
            assert isinstance(probes[stage][loc][name]['data'], ArraySparse)
            data_point = probes[stage][loc][name]['data']
            if stage == _Stage.INPUT:
                sparse_inputs.append(data_point)
            elif stage == _Stage.OUTPUT:
                sparse_outputs.append(data_point)
            else:
                sparse_hints.append(data_point)
        else:  # 如果是dense才进行扩展
            isinstance(probes[stage][loc][name]['data'], ArrayDense)
            dim_to_expand = 1 if stage == _Stage.HINT else 0
            data_point = DataPoint(name=name, location=loc, type_=t,
                                   data=np.expand_dims(data, dim_to_expand))
            if stage == _Stage.INPUT:
                inputs.append(data_point)
            elif stage == _Stage.OUTPUT:
                # outputs.append(data_point)
                raise ProbeError('In YZDDFATasks, there is not any dense output probe.')
            else:
                hints.append(data_point)

    return inputs, hints, sparse_inputs, sparse_outputs, sparse_hints
# pylint: disable=invalid-name
