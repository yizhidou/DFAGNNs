from typing import Optional, Union, List
import os, json, sys, argparse
import numpy as np
from programl.proto import *
import programl
import hashlib
import jax
import jax.numpy as jnp
import signal
import networkx as nx
import matplotlib.pyplot as plt
from clrs._src import dfa_specs, probing

_Array = np.ndarray
taskname_shorts = dict(liveness='l',
                       dominance='d',
                       reachability='r')
np.set_printoptions(threshold=sys.maxsize)


class DFAException(probing.ProbeError):
    # the current sample has been previously recognized as errored
    RECORDED_ERRORED_SAMPLE = 0
    # newly recognized error sample
    ANALYZE_ERRORED_SAMPLE = 1
    # too few interested points error (less than selected num)
    TOO_FEW_IP_NODES = 2
    # too much program points error
    TOO_MANY_PP_NODES = 3

    UNRECOGNIZED_ACTIVATION_TYPE = 4

    UNRECOGNIZED_TASK_NAME = 5

    UNRECOGNIZED_GNN_TYPE = 6

    TOO_LONG_TRACE = 7

    ANALYZE_TIME_OUT = 8

    TOO_FEW_PP_NODES = 9

    NUM_PP_DISAGREE = 10

    def __init__(self, error_code: int,
                 sample_id: Union[str, None] = None):
        self.error_code = error_code
        self.sample_id = sample_id
        super().__init__()

    def error_msg(self):
        if self.error_code == self.RECORDED_ERRORED_SAMPLE:
            msg = 'This sample has previously been recorded as errored!'
        elif self.error_code == self.TOO_FEW_IP_NODES:
            msg = 'This sample has too few IP nodes, so we drop it!'
        elif self.error_code == self.TOO_MANY_PP_NODES:
            msg = 'This sample has too many PP nodes, so we drop it!'
        elif self.error_code == self.UNRECOGNIZED_ACTIVATION_TYPE:
            msg = 'Unrecognized activation type! please check your spelling!'
        elif self.error_code == self.UNRECOGNIZED_TASK_NAME:
            msg = 'Unrecognized task name! please check your spelling!'
        elif self.error_code == self.UNRECOGNIZED_GNN_TYPE:
            msg = 'Unrecognized gnn type!'
        else:
            msg = 'Unrecognized error!'
        if self.sample_id:
            msg += f' sample_id: {self.sample_id}'
        return msg


def timeout_handler(signum, frame):
    raise DFAException(error_code=DFAException.ANALYZE_TIME_OUT)


class SamplePathProcessor:
    def __init__(self, sourcegraph_dir: str,
                 errorlog_savepath: str,
                 # statistics_savepath: Union[str, None] = None
                 ):
        self.sourcegraph_dir = sourcegraph_dir
        self.errorlog_savepath = errorlog_savepath
        if not os.path.isfile(self.errorlog_savepath):
            os.system(f'touch {self.errorlog_savepath}')
            self.errored_sample_ids = {}
        elif os.path.getsize(self.errorlog_savepath) == 0:
            self.errored_sample_ids = {}
        else:
            with open(self.errorlog_savepath) as errored_reader:
                self.errored_sample_ids = json.load(errored_reader)


    def sourcegraph_savepath(self, sample_id):
        return os.path.join(self.sourcegraph_dir, sample_id + '.ProgramGraph.pb')

    def dump_errored_samples_to_log(self):
        with open(self.errorlog_savepath, 'w') as errored_samples_dumper:
            json.dump(self.errored_sample_ids, errored_samples_dumper, indent=3)


class SampleLoader:
    def __init__(self, sample_path_processor: SamplePathProcessor,
                 # max_iteration: int,
                 expected_trace_len: int,
                 cfg_edges_rate: float,
                 selected_num_ip: int,
                 if_sync: bool,
                 seed: int,
                 max_num_pp: Optional[int],  # set None if use it to get statistics
                 min_num_pp: Optional[int],
                 # use_self_loops: bool,
                 trace_sample_from_start: bool,
                 dfa_version: Optional[int],
                 # for_get_statistics: bool = False,
                 if_idx_reorganized: bool = True,
                 selected_ip_fixed: bool = False
                 ):
        self.sample_path_processor = sample_path_processor
        self.expected_trace_len = expected_trace_len
        self.expected_hint_len = self.expected_trace_len - 1
        self.cfg_edges_rate = cfg_edges_rate
        # self.for_get_statistics = for_get_statistics
        self.max_num_pp = max_num_pp
        self.min_num_pp = min_num_pp
        # self.use_self_loops = use_self_loops
        self.trace_sample_from_start = trace_sample_from_start
        # if self.for_get_statistics:
        #     assert self.max_num_pp is None and self.min_num_pp is None
        self.if_sync = if_sync
        if self.if_sync:
            self.max_iteration = 500
        else:
            self.max_iteration = self.expected_trace_len - 1
        self.dfa_version = dfa_version

        self.if_idx_reorganized = if_idx_reorganized
        self.selected_ip_fixed = selected_ip_fixed
        # self.if_record_statistics = if_record_statistics
        self.selected_num_ip = selected_num_ip
        self._rng = np.random.RandomState(seed)

    def _parse_cpp_stdout(self, cpp_out: bytes):
        def _get_a_line(star_idx: int):
            if cpp_out[star_idx:star_idx + 1] == b'\n':
                real_start_idx = star_idx + 1
            else:
                real_start_idx = star_idx
            end_idx = cpp_out[real_start_idx:].find(b'\n')
            line_bytes = cpp_out[real_start_idx: end_idx + real_start_idx]
            return end_idx + real_start_idx, line_bytes

        def _parse_a_line(line: bytes):
            task_name_in_byte, item_name_in_byte, num = line.split(b' ')
            return str(task_name_in_byte, 'utf-8'), str(item_name_in_byte, 'utf-8'), int(num)

        end_idx_be, byte_str_be = _get_a_line(star_idx=0)
        printed_task_name, _, num_be = _parse_a_line(byte_str_be)
        end_idx_pp, byte_str_pp = _get_a_line(star_idx=end_idx_be)
        _, _, num_pp = _parse_a_line(byte_str_pp)
        # sample_statistics['num_pp'] = num_pp
        end_idx_ip, byte_str_ip = _get_a_line(star_idx=end_idx_pp)
        _, _, num_ip = _parse_a_line(byte_str_ip)
        # sample_statistics['num_ip'] = num_ip
        end_idx_it, byte_str_it = _get_a_line(star_idx=end_idx_ip)
        _, item_name, printed_num_iteration = _parse_a_line(byte_str_it)
        end_idx_du, byte_str_du = _get_a_line(star_idx=end_idx_it)
        end_idx_edge_size, byte_str_edge_size = _get_a_line(star_idx=end_idx_du)
        task_name_in_byte, _, edge_size = _parse_a_line(byte_str_edge_size)
        edge_chunck = cpp_out[end_idx_edge_size + 1: end_idx_edge_size + 1 + edge_size]
        trace_chunck = cpp_out[end_idx_edge_size + 1 + edge_size:]
        return num_be, printed_num_iteration, edge_chunck, trace_chunck


    def _load_sparse_trace_from_bytes(self, task_name: Union[str, bytes],
                                      trace_bytes: bytes,
                                      sample_id: str,
                                      selected_num_ip: int,
                                      start_trace_idx: int):
        result_obj = ResultsEveryIteration()
        result_obj.ParseFromString(trace_bytes)
        num_pp = len(result_obj.program_points.value)
        if self.max_num_pp is not None and num_pp > self.max_num_pp:
            self.sample_path_processor.errored_sample_ids[sample_id] = DFAException.TOO_MANY_PP_NODES
            raise DFAException(DFAException.TOO_MANY_PP_NODES, sample_id)
        if self.min_num_pp is not None and num_pp < self.min_num_pp:
            self.sample_path_processor.errored_sample_ids[sample_id] = DFAException.TOO_FEW_PP_NODES
            raise DFAException(DFAException.TOO_FEW_PP_NODES, sample_id)
        num_ip = len(result_obj.interested_points.value)
        if not task_name == 'liveness':
            assert num_pp == num_ip
        if num_ip < selected_num_ip:
            self.sample_path_processor.errored_sample_ids[sample_id] = DFAException.TOO_FEW_IP_NODES
            raise DFAException(DFAException.TOO_FEW_IP_NODES, sample_id)

        if not self.selected_ip_fixed:
            selected_ip_indices_base = np.array(
                sorted(self._rng.choice(a=range(num_ip), size=selected_num_ip, replace=False)))
        else:
            selected_ip_indices_base = np.array([0, 1, 2, 3, 4])
        full_trace_len = len(result_obj.results_every_iteration)  # this should be num_iteration+1
        trace_list = []
        cur_trace_idx = start_trace_idx
        while cur_trace_idx < min(full_trace_len - 1, start_trace_idx + self.expected_trace_len):
            # for trace_idx in range(trace_start_idx, trace_start_idx + self.expected_trace_len):
            trace_content_base = np.zeros(shape=(num_ip, num_pp), dtype=int)
            trace_of_this_iteration = result_obj.results_every_iteration[cur_trace_idx].result_map
            cur_trace_idx += 1
            # generate a matrix from the trace
            assert len(trace_of_this_iteration) == num_pp
            for pp in trace_of_this_iteration.keys():
                active_ip_list = trace_of_this_iteration[pp].value
                num_active_ip_node = len(active_ip_list)
                if num_active_ip_node == 0:
                    continue
                # put corresponding value into trace_matrix
                if task_name == 'liveness':
                    trace_content_base[np.array(active_ip_list) - num_pp, np.repeat(pp, num_active_ip_node)] = 1
                else:
                    trace_content_base[np.array(active_ip_list), np.repeat(pp, num_active_ip_node)] = 1
            trace_content_sparse = trace_content_base[selected_ip_indices_base, :]
            # [selected_num_ip, num_pp]

            trace_list.append(trace_content_sparse.transpose())
            # [num_pp, selected_num_ip]
        # print(f'length of trace_list = {len(trace_list)}')
        return trace_list, selected_ip_indices_base

    def _load_sparse_edge_from_str(self, task_name: Union[str, bytes],
                                   edges_str,
                                   selected_ip_indices_base: np.ndarray):
        edges_saved_matrix = np.fromstring(edges_str, sep=' ', dtype=int)
        assert selected_ip_indices_base.ndim == 1 and selected_ip_indices_base.shape[0] == self.selected_num_ip
        if task_name == 'liveness':
            edges_saved_matrix = edges_saved_matrix.reshape((-1, 3))
            # print(f'the shape of edges_saved_matrix is: {edges_saved_matrix.shape}')
            num_pp, num_ip = edges_saved_matrix[0, 0], edges_saved_matrix[0, 1]
            cfg_row_indices = np.where(edges_saved_matrix[:, -1] == 0)[0]
            gen_row_indices = np.where(edges_saved_matrix[:, -1] == 1)[0]
            kill_row_indices = np.where(edges_saved_matrix[:, -1] == 2)[0]
            cfg_edges = edges_saved_matrix[cfg_row_indices, :-1]
            # num_cfg_edges = cfg_edges.shape[0]
            gen_edges = edges_saved_matrix[gen_row_indices, :-1][:, [1, 0]]
            kill_edges = edges_saved_matrix[kill_row_indices, :-1][:, [1, 0]]

            gen_array_dense = np.zeros(shape=(num_ip, num_pp), dtype=int)
            kill_array_dense = np.zeros(shape=(num_ip, num_pp), dtype=int)
            gen_array_dense[gen_edges[:, 0] - num_pp, gen_edges[:, 1]] = 1
            kill_array_dense[kill_edges[:, 0] - num_pp, kill_edges[:, 1]] = 1

            gen_vectors = gen_array_dense[selected_ip_indices_base, :].transpose()
            # [num_pp, selected_num_ip ]
            kill_vectors = kill_array_dense[selected_ip_indices_base, :].transpose()
            # [num_pp, selected_num_ip, ]
            return num_pp, cfg_edges, [gen_vectors, kill_vectors, _derive_bidirectional_cfg(cfg_indices=cfg_edges,
                                                                                            if_forward=False, )]
            # num_pp=num_pp if self.use_self_loops else None)


        else:
            assert task_name in ['dominance', 'reachability']
            edges_saved_matrix = edges_saved_matrix.reshape((-1, 2))
            num_pp = edges_saved_matrix[0, 0]
            cfg_edges = edges_saved_matrix[1:, :]
            # num_cfg_edges = cfg_edges.shape[0]
            gen_array_dense = np.identity(num_pp, dtype=int)
            gen_vectors = gen_array_dense[selected_ip_indices_base, :].transpose()
            # [num_pp, selected_num_ip]
            kill_vectors = np.zeros((num_pp, self.selected_num_ip), dtype=int)
            # [num_pp, selected_num_ip]
            return num_pp, cfg_edges, [gen_vectors, kill_vectors, _derive_bidirectional_cfg(cfg_indices=cfg_edges,
                                                                                            if_forward=True if task_name == 'dominance' else False)]
            # num_pp=num_pp if self.use_self_loops else None)

    def load_a_sample(self, task_name, sample_id):
        assert task_name in ['liveness', 'dominance', 'reachability']
        if sample_id in self.sample_path_processor.errored_sample_ids:
            raise DFAException(DFAException.RECORDED_ERRORED_SAMPLE, sample_id)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(20)
        try:
            cpp_out, cpperror = programl.yzd_analyze(task_name=_get_analyze_task_name(task_name),
                                                     max_iteration=self.max_iteration,
                                                     program_graph_sourcepath=self.sample_path_processor.sourcegraph_savepath(
                                                         sample_id),
                                                     edge_list_savepath=None,
                                                     result_savepath=None,
                                                     if_sync=self.if_sync,
                                                     if_idx_reorganized=self.if_idx_reorganized)
        except DFAException as e:
            print(f'error happens during analyze! error_code = {e.error_code}')
            self.sample_path_processor.errored_sample_ids[sample_id] = DFAException.ANALYZE_TIME_OUT
            raise e
        signal.alarm(0)
        if len(cpperror) > 0:
            if cpperror.endswith(b'in certain iterations!\n'):
                print(f'cpp reports that the trace length exceeds {self.max_iteration}!')
                self.sample_path_processor.errored_sample_ids[sample_id] = DFAException.TOO_LONG_TRACE
                raise DFAException(DFAException.TOO_LONG_TRACE, sample_id)
            else:
                print('cpp reports an error!')
                self.sample_path_processor.errored_sample_ids[sample_id] = DFAException.ANALYZE_ERRORED_SAMPLE
                raise DFAException(DFAException.ANALYZE_ERRORED_SAMPLE, sample_id)
        _, printed_trace_len, edge_chunck, trace_chunck = self._parse_cpp_stdout(cpp_out=cpp_out)
        # if self.sample_path_processor.statistics_savepath:
        #     self._merge_statistics(sample_id, sample_statistics)
        # print(f'dfa_utils line 359, the real_trace_len = {printed_trace_len}')
        if self.if_sync and printed_trace_len > self.expected_trace_len and not self.trace_sample_from_start:
            trace_start_idx = self._rng.randint(0, printed_trace_len - self.expected_trace_len)
        else:
            trace_start_idx = 0
        trace_list, selected_ip_indices_base = self._load_sparse_trace_from_bytes(task_name=task_name,
                                                                                  trace_bytes=trace_chunck,
                                                                                  sample_id=sample_id,
                                                                                  selected_num_ip=self.selected_num_ip,
                                                                                  start_trace_idx=trace_start_idx)
        num_pp, cfg_edges, array_list = self._load_sparse_edge_from_str(task_name=task_name,
                                                                        edges_str=edge_chunck,
                                                                        selected_ip_indices_base=selected_ip_indices_base)
        return trace_list, array_list, printed_trace_len


    def _num_pp_and_cfg(self, task_name: Union[str, bytes],
                        edges_str):
        edges_saved_matrix = np.fromstring(edges_str, sep=' ', dtype=int)
        if task_name == 'liveness':
            edges_saved_matrix = edges_saved_matrix.reshape((-1, 3))
            num_pp, num_ip = edges_saved_matrix[0, 0], edges_saved_matrix[0, 1]
            cfg_row_indices = np.where(edges_saved_matrix[:, -1] == 0)[0]
            cfg_edges = edges_saved_matrix[cfg_row_indices, :-1]
        else:
            assert task_name in ['dominance', 'reachability']
            edges_saved_matrix = edges_saved_matrix.reshape((-1, 2))
            num_pp = edges_saved_matrix[0, 0]
            cfg_edges = edges_saved_matrix[1:, :]
        return num_pp, cfg_edges

    def get_statistics(self, task_name, sample_id):
        assert task_name in ['liveness', 'dominance', 'reachability']
        if sample_id in self.sample_path_processor.errored_sample_ids:
            raise DFAException(DFAException.RECORDED_ERRORED_SAMPLE, sample_id)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)
        try:
            cpp_out, cpperror = programl.yzd_analyze(task_name=_get_analyze_task_name(task_name),
                                                     max_iteration=self.max_iteration,
                                                     program_graph_sourcepath=self.sample_path_processor.sourcegraph_savepath(
                                                         sample_id),
                                                     edge_list_savepath=None,
                                                     result_savepath=None,
                                                     if_sync=self.if_sync,
                                                     if_idx_reorganized=self.if_idx_reorganized)
        except DFAException as e:
            print(f'error happens during analyze! error_code = {e.error_code}')
            self.sample_path_processor.errored_sample_ids[sample_id] = DFAException.ANALYZE_TIME_OUT
            raise e
        signal.alarm(0)
        if len(cpperror) > 0:
            if cpperror.endswith(b'in certain iterations!\n'):
                print(f'cpp reports that the trace length exceeds {self.max_iteration}!')
                self.sample_path_processor.errored_sample_ids[sample_id] = DFAException.TOO_LONG_TRACE
                raise DFAException(DFAException.TOO_LONG_TRACE, sample_id)
            else:
                print('cpp reports an error!')
                self.sample_path_processor.errored_sample_ids[sample_id] = DFAException.ANALYZE_ERRORED_SAMPLE
                raise DFAException(DFAException.ANALYZE_ERRORED_SAMPLE, sample_id)
        num_be, printed_trace_len, edge_chunck, trace_chunck = self._parse_cpp_stdout(cpp_out=cpp_out)
        num_pp, cfg_edges = self._num_pp_and_cfg(task_name=task_name,
                                                 edges_str=edge_chunck)
        # assert num_pp_from_trace == num_pp
        num_cfg_edges = cfg_edges.shape[0]
        # print('dfa_util line 464, num_pp_from_trace is redundant if the assertion passes!')
        if task_name == 'liveness' or task_name == 'reachability':
            cfg_edges = cfg_edges[:, [1, 0]]
        else:
            assert task_name == 'dominance'
        cfg_ad_matrix = np.zeros((num_pp, num_pp), dtype=int)
        cfg_ad_matrix[cfg_edges[:, 0], cfg_edges[:, 1]] = 1
        out_degree_vector = np.sum(cfg_ad_matrix, axis=-1)
        in_degree_vector = np.sum(cfg_ad_matrix, axis=0)
        max_out_degree = np.max(out_degree_vector).item()
        max_in_degree = np.max(in_degree_vector).item()
        return num_pp, num_cfg_edges, num_be, max_out_degree, max_in_degree, printed_trace_len

    def visualize_the_sample(self, sample_id,
                             savepath: Optional[str] = None):
        if sample_id in self.sample_path_processor.errored_sample_ids:
            raise DFAException(DFAException.RECORDED_ERRORED_SAMPLE, sample_id)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)
        try:
            cpp_out, cpperror = programl.yzd_analyze(task_name=_get_analyze_task_name('dominance'),
                                                     max_iteration=self.max_iteration,
                                                     program_graph_sourcepath=self.sample_path_processor.sourcegraph_savepath(
                                                         sample_id),
                                                     edge_list_savepath=None,
                                                     result_savepath=None,
                                                     if_sync=self.if_sync,
                                                     if_idx_reorganized=self.if_idx_reorganized)
        except DFAException as e:
            print(f'error happens during analyze! error_code = {e.error_code}')
            self.sample_path_processor.errored_sample_ids[sample_id] = DFAException.ANALYZE_TIME_OUT
            raise e
        signal.alarm(0)
        if len(cpperror) > 0:
            if cpperror.endswith(b'in certain iterations!\n'):
                print(f'cpp reports that the trace length exceeds {self.max_iteration}!')
                self.sample_path_processor.errored_sample_ids[sample_id] = DFAException.TOO_LONG_TRACE
                raise DFAException(DFAException.TOO_LONG_TRACE, sample_id)
            else:
                print('cpp reports an error!')
                self.sample_path_processor.errored_sample_ids[sample_id] = DFAException.ANALYZE_ERRORED_SAMPLE
                raise DFAException(DFAException.ANALYZE_ERRORED_SAMPLE, sample_id)
        _, printed_trace_len, edge_chunck, trace_chunck = self._parse_cpp_stdout(cpp_out=cpp_out)
        edges_saved_matrix = np.fromstring(edge_chunck, sep=' ', dtype=int)
        edges_saved_matrix = edges_saved_matrix.reshape((-1, 2))
        num_pp = edges_saved_matrix[0, 0]
        cfg_edges = edges_saved_matrix[1:, :]
        G = nx.Graph()
        G.add_nodes_from(range(num_pp))
        G.add_edges_from(cfg_edges)
        nx.draw_networkx(G, arrows=True, arrowstyle='-|>', arrowsize=12)
        if savepath is not None:
            plt.savefig(fname=savepath, format='pdf')


def _derive_bidirectional_cfg(cfg_indices,
                              if_forward,
                              # num_pp: Optional[int]
                              ):
    # if num_pp is not None, self-loops are added
    num_cfg_edges = cfg_indices.shape[0]
    dual_cfg_indices = cfg_indices[:, [1, 0]]
    cfg_edges_forward = np.concatenate([cfg_indices if if_forward else dual_cfg_indices,
                                        np.ones((num_cfg_edges, 1), dtype=int)],
                                       axis=1)
    # [num_cfg, 3]
    cfg_edges_backward = np.concatenate([cfg_indices if not if_forward else dual_cfg_indices,
                                         np.zeros((num_cfg_edges, 1), dtype=int)],
                                        axis=1)
    # if num_pp is None:
    return np.concatenate([cfg_edges_forward, cfg_edges_backward], axis=0)
    #   [2*num_cfg, 3]
    # else:
    #     self_loops_indices = np.repeat(np.arange(num_pp).reshape((num_pp, 1)),
    #                                    axis=1, repeats=2)
    #     #   [num_pp, 2]
    #     self_loops_content = 2 * np.ones((num_pp, 1), dtype=int)
    #     #   [num_pp, 1]
    #     self_loops = np.concatenate([self_loops_indices, self_loops_content], axis=1)
    #     #   [num_pp, 3]
    #     return np.concatenate([cfg_edges_forward, cfg_edges_backward, self_loops], axis=0)
    #     #   [2*num_cfg + num_pp]


def _get_analyze_task_name(task_name: str):
    if task_name == 'liveness':
        return 'yzd_liveness'
    elif task_name == 'reachability':
        return 'yzd_reachability'
    elif task_name == 'dominance':
        return 'yzd_dominance'
    return task_name


def _get_activation(activation_str):
    if activation_str == 'relu':
        return jax.nn.relu
    raise DFAException(DFAException.UNRECOGNIZED_ACTIVATION_TYPE)


def dim_expand_to(x, y):
    while len(y.shape) > len(x.shape):
        x = jnp.expand_dims(x, -1)
    return x


def filter_sample_list(full_statistics_savepath,
                       errored_sample_ids,
                       max_num_pp,
                       min_num_pp,
                       cfg_edges_rate,
                       # sample_id_savepath: Union[str, List],
                       sample_ids: List
                       ):
    assert min_num_pp < max_num_pp
    with open(full_statistics_savepath) as statistics_loader:
        full_statistics_dict = json.load(statistics_loader)
    filtered_sample_ids = []
    for sample_id in sample_ids:
        if sample_id in full_statistics_dict:
            num_pp, num_cfg = full_statistics_dict[sample_id][:2]
            if num_pp > max_num_pp or num_pp < min_num_pp:
                continue
            if num_cfg > max_num_pp * cfg_edges_rate:
                continue
        else:
            # try:
            assert sample_id in errored_sample_ids
            continue
        filtered_sample_ids.append(sample_id)
    return filtered_sample_ids



def new_get_statistics_from_dataset(sourcegraph_dir: str,
                                    errorlog_savepath: str,
                                    sample_ids_savepath: str,
                                    num_pp_statistics_log_savepath: Optional[str],
                                    full_statistics_log_savepath: Optional[str],
                                    if_clear_num_pp_statistics: bool,
                                    if_clear_full_statistics):
    sample_path_processor = SamplePathProcessor(
        sourcegraph_dir=sourcegraph_dir,
        errorlog_savepath=errorlog_savepath)
    sample_loader = SampleLoader(sample_path_processor=sample_path_processor,
                                 expected_trace_len=6,
                                 max_num_pp=None,
                                 min_num_pp=None,
                                 cfg_edges_rate=1.5,
                                 selected_num_ip=5,
                                 # for_get_statistics=True,
                                 # dfa_version=0,
                                 # use_self_loops=True,
                                 if_sync=True,
                                 trace_sample_from_start=True,
                                 seed=6)
    if num_pp_statistics_log_savepath is not None:
        if if_clear_num_pp_statistics or (not os.path.isfile(num_pp_statistics_log_savepath)) or os.path.getsize(
                num_pp_statistics_log_savepath) == 0:
            num_pp_statistics_dict = {}
        else:
            with open(num_pp_statistics_log_savepath) as f:
                num_pp_statistics_dict = json.load(f)
    else:
        num_pp_statistics_dict = {}
    if full_statistics_log_savepath is not None:
        if if_clear_full_statistics or (not os.path.isfile(full_statistics_log_savepath)) or os.path.getsize(
                full_statistics_log_savepath) == 0:
            full_statistics_dict = {}
        else:
            with open(full_statistics_log_savepath) as f:
                full_statistics_dict = json.load(f)
    else:
        full_statistics_dict = {}
    count = 0
    with open(sample_ids_savepath) as f:
        for line in f.readlines():
            sample_id = line.strip()
            if sample_id in sample_path_processor.errored_sample_ids or (
                    sample_id in num_pp_statistics_dict and sample_id in full_statistics_dict):
                print(f'{sample_id} has been processed, so skip!')
                continue
            count += 1
            if count % 500 == 0:
                sample_path_processor.dump_errored_samples_to_log()
                if num_pp_statistics_log_savepath is not None:
                    with open(num_pp_statistics_log_savepath, 'w') as pp_statistics_logger:
                        json.dump(num_pp_statistics_dict, pp_statistics_logger, indent=3)
                if full_statistics_log_savepath is not None:
                    with open(full_statistics_log_savepath, 'w') as full_statistics_logger:
                        json.dump(full_statistics_dict, full_statistics_logger, indent=3)
            print(f'{count}: {sample_id} id on processing...')
            # num_pp = None
            try:
                num_pp_liveness, num_cfg_liveness, num_be_liveness, max_out_degree_liveness, max_in_degree_liveness, printed_trace_len_liveness = sample_loader.get_statistics(
                    task_name='liveness',
                    sample_id=sample_id)
                num_pp_reachability, num_cfg_reachability, num_be_reachability, max_out_degree_reachability, max_in_degree_reachability, printed_trace_len_reachability = sample_loader.get_statistics(
                    task_name='reachability',
                    sample_id=sample_id)
                num_pp_dominance, num_cfg_dominance, num_be_dominance, max_out_degree_dominance, max_in_degree_dominance, printed_trace_len_dominance = sample_loader.get_statistics(
                    task_name='dominance',
                    sample_id=sample_id)
                try:
                    assert num_pp_liveness == num_pp_reachability and num_pp_liveness == num_pp_dominance
                except AssertionError:
                    print('dfa_utils line 820, 3 num_pps disagree!')
                    print(
                        f'num_pp_liveness = {num_pp_liveness}; num_pp_reachability = {num_pp_reachability}; num_pp_dominance = {num_pp_dominance}')
                    sample_path_processor.errored_sample_ids[sample_id] = DFAException.NUM_PP_DISAGREE
                    raise DFAException(DFAException.NUM_PP_DISAGREE)
                assert num_cfg_liveness == num_cfg_reachability and num_cfg_liveness == num_cfg_dominance
                assert max_out_degree_liveness == max_out_degree_reachability and max_out_degree_liveness == max_out_degree_dominance
                assert max_in_degree_liveness == max_in_degree_reachability and max_in_degree_liveness == max_in_degree_dominance
                # assert num_be_liveness == num_be_reachability, f'num_be_l = {num_be_liveness}; num_be_r = {num_be_reachability}'
                # num_be_backeard = num_be_liveness
                # num_be_forward = num_be_dominance
                num_pp = num_pp_liveness.item()
                num_cfg = num_cfg_liveness
                max_out_degree = max_out_degree_liveness
                max_in_degree = max_in_degree_liveness
                num_pp_statistics_dict[sample_id] = num_pp
                # en_ratio = float(num_cfg) / float(num_pp)
                full_statistics_dict[sample_id] = (
                    num_pp, num_cfg, num_be_liveness, num_be_reachability, num_be_dominance, max_out_degree,
                    max_in_degree, printed_trace_len_liveness,
                    printed_trace_len_reachability, printed_trace_len_dominance)
                print(
                    f'success! num_pp = {num_pp_liveness}; num_cfg = {num_cfg_liveness}; num_be_l = {num_be_liveness}; num_be_r = {num_be_reachability}; num_be_d = {num_be_dominance}; max_out = {max_out_degree_liveness}; max_in = {max_in_degree_liveness}; tl_l = {printed_trace_len_liveness}; tl_r = {printed_trace_len_reachability}; tl_d = {printed_trace_len_dominance}')
            except DFAException as e:
                print(f'{sample_id} errored! error_code: {e.error_code}')
                continue
            # break
    sample_path_processor.dump_errored_samples_to_log()
    if num_pp_statistics_log_savepath is not None:
        with open(num_pp_statistics_log_savepath, 'w') as pp_statistics_logger:
            json.dump(num_pp_statistics_dict, pp_statistics_logger, indent=3)
    if full_statistics_log_savepath is not None:
        with open(full_statistics_log_savepath, 'w') as full_statistics_logger:
            json.dump(full_statistics_dict, full_statistics_logger, indent=3)


def compute_hash(file_path,
                 # if_for_test: bool
                 ):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read and update hash in chunks of 4K
        for line in f:
            flag = 1
            if flag:
                sha256_hash.update(line)
    return sha256_hash.hexdigest()


class UtilPathProcessor:
    AVAILABLE_DATASET_NAMES = ['poj104', 'tensorflow', 'linux', 'opencv', 'opencl', 'npd']

    def dataset_sample_ids_savepath(self, dataset_name):
        return f'{dataset_name}/{dataset_name}_sample_ids.txt'

    def trained_model_params_savedir(self, dataset_name):
        # assert dataset_name in self.AVAILABLE_DATASET_NAMES
        return f'{dataset_name}_TrainParams'

    def train_log_savepath(self, dataset_name, params_hash):
        log_savedir = f'{dataset_name}_TrainLogs'
        os.path.join(log_savedir, f'{params_hash}.log')

    def ckpt_savedir(self, dataset_name, params_hash):
        # assert dataset_name in self.AVAILABLE_DATASET_NAMES
        savedir = ''
        return os.path.join(savedir, f'{dataset_name}_CKPT', f'{params_hash}_ckpt')

    def test_sample_ids_savepath(self, dataset_name):
        return f'{dataset_name}/{dataset_name}_sample_ids.txt'

    def test_info_savedir(self, dataset_name):
        # assert dataset_name in self.AVAILABLE_DATASET_NAMES
        return f'{dataset_name}_TestInfo'

    def test_log_savedir(self, dataset_name, trained_model_params_id, test_info_hash):
        # assert dataset_name in self.AVAILABLE_DATASET_NAMES
        savedir = f'{dataset_name}_TestLogs'
        return os.path.join(savedir, f'{trained_model_params_id}_trained', f'{test_info_hash}_test')

    def num_pp_statistics_filepath(self, dataset_name):
        # assert dataset_name in self.AVAILABLE_DATASET_NAMES
        return f'{dataset_name}_Statistics/{dataset_name}_num_pp_statistics.json'

    def full_statistics_filepath(self, dataset_name):
        # assert dataset_name in self.AVAILABLE_DATASET_NAMES
        return f'{dataset_name}_Statistics/{dataset_name}_full_statistics.json'

    def errorlog_savepath(self, dataset_name):
        return f'{dataset_name}_errors_max500.txt'

    def case_analysis_savepath(self, dataset_name, test_info_id, ckpt_idx):
        return f'{dataset_name}_CaseAnalysis/{test_info_id}_case_analysis/{test_info_id}_ckpt{ckpt_idx}.analysis.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Please input the params filename')
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--clear_pp', action='store_true')
    parser.add_argument('--clear_full', action='store_true')
    args = parser.parse_args()
    assert args.dataset_name in ['poj104', 'github', 'tensorflow', 'linux']
    sourcegraph_dir = ''
    errorlog_savepath = ''
    util_path_processor = UtilPathProcessor()
    new_get_statistics_from_dataset(sourcegraph_dir=sourcegraph_dir,
                                    errorlog_savepath=errorlog_savepath,
                                    sample_ids_savepath=util_path_processor.dataset_sample_ids_savepath(
                                        args.dataset_name),
                                    num_pp_statistics_log_savepath=util_path_processor.num_pp_statistics_filepath(
                                        args.dataset_name),
                                    full_statistics_log_savepath=util_path_processor.full_statistics_filepath(
                                        args.dataset_name),
                                    if_clear_num_pp_statistics=args.clear_pp,
                                    if_clear_full_statistics=args.clear_full)
