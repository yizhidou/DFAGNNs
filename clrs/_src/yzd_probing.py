from clrs._src.probing import *

_Location = specs.Location
_Stage = specs.Stage
_Type = specs.Type
_ArraySparse = collections.namedtuple('ArraySparse', ['edges', 'nb_nodes', 'nb_edges'])
_ArrayDense = np.ndarray
_Array = Union[_ArrayDense, _ArraySparse]


def yzd_finalize(probes: ProbesDict):
    """Finalizes a `ProbesDict` by stacking/squeezing `data` field."""
    for stage in [_Stage.INPUT, _Stage.OUTPUT, _Stage.HINT]:
        for loc in [_Location.NODE, _Location.EDGE, _Location.GRAPH]:
            for name in probes[stage][loc]:
                if isinstance(probes[stage][loc][name]['data'], _ArraySparse) or isinstance(
                        probes[stage][loc][name]['data'], _ArrayDense):
                    raise ProbeError('Attemping to re-finalize a finalized `ProbesDict`.')
                if stage == _Stage.HINT:
                    # Hints are provided for each timestep. Stack them here.
                    if loc == _Location.EDGE:
                        edges_list = []
                        nb_edges_list = []
                        nb_nodes = probes[stage][loc][name]['data'][0].nb_nodes
                        for hint_idx, dp in enumerate(probes[stage][loc][name]['data']):
                            assert len(dp.edges.shape) == 2 and dp.edges.shape[-1] == 2
                            edges_list.append(dp.edges)
                            assert isinstance(dp.nb_edges, int)
                            nb_edges_list.append(dp.nb_edges)
                            assert dp.nb_nodes == nb_nodes
                        probes[stage][loc][name]['data'] = _ArraySparse(edges=np.concatenate(edges_list, axis=0),
                                                                        # [nb_edges_total, 2]
                                                                        nb_nodes=nb_nodes,
                                                                        nb_edges=np.expand_dims(np.array(nb_edges_list),
                                                                                                0)  # [1, hint_len]
                                                                        )

                    else:
                        probes[stage][loc][name]['data'] = np.stack(
                            probes[stage][loc][name]['data'])
                else:
                    # Only one instance of input/output exist. Remove leading axis.
                    assert len(probes[stage][loc][name]['data']) == 1
                    if loc == _Location.EDGE:
                        assert isinstance(probes[stage][loc][name]['data'][0], _ArraySparse)
                        probes[stage][loc][name]['data'] = _ArraySparse(edges=probes[stage][loc][name]['data'].edges,
                                                                        nb_nodes=probes[stage][loc][name][
                                                                            'data'].nb_nodes,
                                                                        nb_edges=np.array([[probes[stage][loc][name][
                                                                                                'data'].nb_edges]])
                                                                        # [1, 1]
                                                                        )
                    else:
                        assert isinstance(probes[stage][loc][name]['data'][0], _ArrayDense)
                        probes[stage][loc][name]['data'] = np.squeeze(
                            np.array(probes[stage][loc][name]['data']))


def yzd_split_stages(probes: ProbesDict,
                     spec: specs.Spec):
    """Splits contents of `ProbesDict` into `DataPoint`s by stage."""

    inputs = []
    outputs = []
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
        if not isinstance(probes[stage][loc][name]['data'], _ArraySparse) and not isinstance(
                probes[stage][loc][name]['data'], _ArrayDense):
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

        # 如果是dense才扩展
        if isinstance(probes[stage][loc][name]['data'], _ArrayDense):
            dim_to_expand = 1 if stage == _Stage.HINT else 0
            data_point = DataPoint(name=name, location=loc, type_=t,
                                   data=np.expand_dims(data, dim_to_expand))
            if stage == _Stage.INPUT:
                inputs.append(data_point)
            elif stage == _Stage.OUTPUT:
                outputs.append(data_point)
            else:
                hints.append(data_point)
        else:
            assert isinstance(probes[stage][loc][name]['data'], _ArraySparse)
            data_point = probes[stage][loc][name]['data']
            if stage == _Stage.INPUT:
                sparse_inputs.append(data_point)
            elif stage == _Stage.OUTPUT:
                sparse_outputs.append(data_point)
            else:
                sparse_hints.append(data_point)

    return inputs, outputs, hints, sparse_inputs, sparse_outputs, sparse_hints
# pylint: disable=invalid-name