from typing import List, Union, Dict
import os, json, sys
import numpy as np
from programl.proto import *
import programl
import random
import jax
import jax.numpy as jnp
from clrs._src import dfa_specs, probing

_Array = np.ndarray
taskname_shorts = dict(yzd_liveness='yl', yzd_dominance='yd', yzd_reachability='yr')
np.set_printoptions(threshold=sys.maxsize)


# class YZDExcpetion(probing.ProbeError):
#     # the current sample has been previously recognized as errored
#     RECORDED_ERRORED_SAMPLE = 0
#     # newly recognized error sample
#     ANALYZE_ERRORED_SAMPLE = 1
#     # too few interested points error (less than selected num)
#     TOO_FEW_IP_NODES = 2
#     # too much program points error
#     TOO_MANY_PP_NODES = 3
#
#     UNRECOGNIZED_ACTIVATION_TYPE = 4
#
#     UNRECOGNIZED_TASK_NAME = 5
#
#     def __init__(self, error_code: int,
#                  sample_id: Union[str, None] = None):
#         self.error_code = error_code
#         self.sample_id = sample_id
#         super().__init__()
#
#     def error_msg(self):
#         if self.error_code == self.RECORDED_ERRORED_SAMPLE:
#             msg = 'This sample has previously been recorded as errored!'
#         # elif self.error_code == self.NEWLY_ERRORED_SAMPLE:
#         #     msg = 'This sample is newly recognized errored, we have now recorded it!'
#         elif self.error_code == self.TOO_FEW_IP_NODES:
#             msg = 'This sample has too few IP nodes, so we drop it!'
#         elif self.error_code == self.TOO_MANY_PP_NODES:
#             msg = 'This sample has too many PP nodes, so we drop it!'
#         elif self.error_code == self.UNRECOGNIZED_ACTIVATION_TYPE:
#             msg = 'Unrecognized activation type! please check your spelling!'
#         elif self.error_code == self.UNRECOGNIZED_TASK_NAME:
#             msg = 'Unrecognized task name! please check your spelling!'
#         else:
#             msg = 'Unrecognized error!'
#         if self.sample_id:
#             msg += f' sample_id: {self.sample_id}'
#         return msg
#

# class SamplePathProcessor:
#     def __init__(self, sourcegraph_dir: str,
#                  errorlog_savepath: str,
#                  dataset_savedir: Union[None, str] = None,
#                  statistics_savepath: Union[None, str] = None):
#         if not os.path.isdir(sourcegraph_dir):
#             # YZDTODO raise an error
#             pass
#         self.sourcegraph_dir = sourcegraph_dir
#         self.errored_sample_ids = {}
#         if not os.path.isfile(errorlog_savepath):
#             os.system(f'touch {errorlog_savepath}')
#         else:
#             with open(errorlog_savepath) as errored_reader:
#                 for line in errored_reader.readlines():
#                     errored_sample_id = line.split(':')[0].strip()
#                     self.errored_sample_ids[errored_sample_id] = 1
#         self.errorlog_savepath = errorlog_savepath
#         self.dataset_savedir = dataset_savedir
#         self.statistics_savepath = statistics_savepath
#
#     def sourcegraph_savepath(self, sample_id):
#         return os.path.join(self.sourcegraph_dir, sample_id + '.ProgramGraph.pb')
#
#     def _trace_savedir(self, task_name):
#         return os.path.join(self.dataset_savedir, task_name, 'Traces')
#
#     def _edge_savedir(self, task_name):
#         return os.path.join(self.dataset_savedir, task_name, 'Edges')
#
#     def trace_savepath(self, task_name, if_sync, sample_id):
#         tmp_str = "sync" if if_sync else "async"
#         return os.path.join(self._trace_savedir(task_name), tmp_str,
#                             sample_id + f'.{taskname_shorts[task_name]}.{tmp_str}.trace')
#
#     def edge_savepath(self, task_name, sample_id):
#         return os.path.join(self._edge_savedir(task_name), sample_id + f'.{taskname_shorts[task_name]}.edge')
#
#     def if_sample_exists(self, task_name, if_syn, sample_id):
#         if not self.dataset_savedir:
#             return False
#         trace_path_to_check = self.trace_savepath(task_name, if_syn, sample_id)
#         edge_path_to_check = self.edge_savepath(task_name, sample_id)
#         if not os.path.isfile(trace_path_to_check) or not os.path.isfile(edge_path_to_check):
#             return False
#         if os.path.getsize(trace_path_to_check) == 0 or os.path.getsize(edge_path_to_check) == 0:
#             return False
#         return True
#

class SampleLoader:
    def __init__(self, sample_path_processor: SamplePathProcessor,
                 # max_iteration: int,
                 expected_trace_len: int,
                 max_num_pp: int,
                 cfg_edges_rate: int,
                 selected_num_ip: int,
                 if_sync: bool,
                 if_idx_reorganized: bool = True,
                 if_save: bool = False
                 ):
        self.sample_path_processor = sample_path_processor
        self.expected_trace_len = expected_trace_len
        self.expected_hint_len = self.expected_trace_len - 1
        self.cfg_edges_rate = cfg_edges_rate
        self.max_num_pp = max_num_pp
        self.if_sync = if_sync
        if self.if_sync:
            self.max_iteration = 200
        else:
            self.max_iteration = self.expected_trace_len - 1
        self.if_idx_reorganized = if_idx_reorganized
        self.if_save = if_save
        self.selected_num_ip = selected_num_ip
        if not self.sample_path_processor.dataset_savedir:
            self.if_save = False
        if self.sample_path_processor.statistics_savepath:
            if not os.path.isfile(self.sample_path_processor.statistics_savepath):
                os.system(f'touch {self.sample_path_processor.statistics_savepath}')
                self.statistics_dict = {}
            elif os.path.getsize(self.sample_path_processor.statistics_savepath) == 0:
                self.statistics_dict = {}
            else:
                with open(self.sample_path_processor.statistics_savepath) as log_reader:
                    self.statistics_dict = json.load(log_reader)

    def load_a_sample(self, task_name, sample_id):
        if sample_id in self.sample_path_processor.errored_sample_ids:
            raise YZDExcpetion(YZDExcpetion.RECORDED_ERRORED_SAMPLE, sample_id)
        if self.sample_path_processor.if_sample_exists(task_name=task_name, if_syn=self.if_sync, sample_id=sample_id):
            trace_savepath = self.sample_path_processor.trace_savepath(task_name, self.if_sync, sample_id)
            with open(trace_savepath, 'rb') as result_reader:
                trace_bytes_from_file = result_reader.read()
            edge_savepath = self.sample_path_processor.edge_savepath(task_name, sample_id)
            with open(edge_savepath) as edges_reader:
                edges_bytes_from_file = edges_reader.read()
            trace_list, selected_ip_indices_base, num_pp = self._load_sparse_trace_from_bytes(task_name=task_name,
                                                                                              trace_bytes=trace_bytes_from_file,
                                                                                              sample_id=sample_id,
                                                                                              selected_num_ip=self.selected_num_ip)
            array_list = self._load_sparse_edge_from_str(task_name=task_name,
                                                         edges_str=edges_bytes_from_file,
                                                         selected_ip_indices_base=selected_ip_indices_base)
        else:
            cpp_out, cpperror = programl.yzd_analyze(task_name=_get_analyze_task_name(task_name),
                                                     max_iteration=self.max_iteration,
                                                     program_graph_sourcepath=self.sample_path_processor.sourcegraph_savepath(
                                                         sample_id),
                                                     edge_list_savepath=self.sample_path_processor.edge_savepath(
                                                         task_name, sample_id) if self.if_save else None,
                                                     result_savepath=self.sample_path_processor.trace_savepath(
                                                         task_name, sample_id) if self.if_save else None,
                                                     if_sync=self.if_sync,
                                                     if_idx_reorganized=self.if_idx_reorganized)
            if len(cpperror) > 0:
                print('new error occurs from cpp! (dfa_utils line 164)')
                print(cpperror)
                self.sample_path_processor.errored_sample_ids[sample_id] = 1
                with open(self.sample_path_processor.errorlog_savepath, 'a') as error_sample_writer:
                    error_sample_writer.write(f'{sample_id}: {YZDExcpetion.ANALYZE_ERRORED_SAMPLE}\n')
                raise YZDExcpetion(YZDExcpetion.ANALYZE_ERRORED_SAMPLE, sample_id)
            sample_statistics, printed_trace_len, edge_chunck, trace_chunck = self._parse_cpp_stdout(cpp_out)
            if self.sample_path_processor.statistics_savepath:
                self._merge_statistics(sample_id, sample_statistics)
            if self.if_sync and printed_trace_len + 1 > self.expected_trace_len:
                trace_start_idx = random.randint(0, printed_trace_len - self.expected_trace_len)
                # print(f'dfa_utils line 183, trace_start_idx = {trace_start_idx}')
            else:
                trace_start_idx = 0
            trace_list, selected_ip_indices_base, num_pp = self._load_sparse_trace_from_bytes(task_name=task_name,
                                                                                              trace_bytes=trace_chunck,
                                                                                              sample_id=sample_id,
                                                                                              selected_num_ip=self.selected_num_ip,
                                                                                              start_trace_idx=trace_start_idx)
            array_list = self._load_sparse_edge_from_str(task_name=task_name,
                                                         edges_str=edge_chunck,
                                                         selected_ip_indices_base=selected_ip_indices_base)
        return trace_list, array_list

    def _merge_statistics(self, sample_id, new_statistics: Dict):
        if not sample_id in self.statistics_dict:
            self.statistics_dict[sample_id] = new_statistics
        else:
            for k, v in new_statistics.items():
                if not k in self.statistics_dict[sample_id]:
                    self.statistics_dict[sample_id][k] = v

    def log_statistics_to_file(self):
        with open(self.sample_path_processor.statistics_savepath, 'w') as writer:
            json.dump(self.statistics_dict, writer, indent=3)

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

        sample_statistics = dict()
        end_idx_be, byte_str_be = _get_a_line(star_idx=0)
        task_name, _, num_be = _parse_a_line(byte_str_be)
        if task_name == 'yzd_dominance':
            sample_statistics['num_be(forward)'] = num_be
        else:
            assert task_name == 'yzd_liveness' or task_name == 'yzd_reachability'
            sample_statistics['num_be(backward)'] = num_be
        end_idx_pp, byte_str_pp = _get_a_line(star_idx=end_idx_be)
        _, _, num_pp = _parse_a_line(byte_str_pp)
        sample_statistics['num_pp'] = num_pp
        end_idx_ip, byte_str_ip = _get_a_line(star_idx=end_idx_pp)
        _, _, num_ip = _parse_a_line(byte_str_ip)
        sample_statistics['num_ip'] = num_ip
        end_idx_it, byte_str_it = _get_a_line(star_idx=end_idx_ip)
        _, item_name, num_iteration = _parse_a_line(byte_str_it)
        sample_statistics[f'{item_name}_{task_name}'] = num_iteration
        printed_trace_len = int(num_iteration)
        end_idx_du, byte_str_du = _get_a_line(star_idx=end_idx_it)
        end_idx_edge_size, byte_str_edge_size = _get_a_line(star_idx=end_idx_du)
        task_name_in_byte, _, edge_size = _parse_a_line(byte_str_edge_size)
        edge_chunck = cpp_out[end_idx_edge_size + 1: end_idx_edge_size + 1 + edge_size]
        trace_chunck = cpp_out[end_idx_edge_size + 1 + edge_size:]
        return sample_statistics, printed_trace_len, edge_chunck, trace_chunck

    def _load_sparse_trace_from_bytes(self, task_name: Union[str, bytes],
                                      trace_bytes: bytes,
                                      sample_id: str,
                                      selected_num_ip: int,
                                      start_trace_idx: int = 0):
        result_obj = ResultsEveryIteration()
        result_obj.ParseFromString(trace_bytes)
        num_pp = len(result_obj.program_points.value)
        if num_pp > self.max_num_pp:
            self.sample_path_processor.errored_sample_ids[sample_id] = 1
            with open(self.sample_path_processor.errorlog_savepath, 'a') as error_sample_writer:
                error_sample_writer.write(f'{sample_id}: {YZDExcpetion.TOO_MANY_PP_NODES}\n')
            raise YZDExcpetion(YZDExcpetion.TOO_MANY_PP_NODES, sample_id)
        num_ip = len(result_obj.interested_points.value)
        if not (task_name == 'dfa_liveness' or task_name == b'dfa_liveness'):
            assert num_pp == num_ip
        if num_ip < selected_num_ip:
            self.sample_path_processor.errored_sample_ids[sample_id] = 1
            with open(self.sample_path_processor.errorlog_savepath, 'a') as error_sample_writer:
                error_sample_writer.write(f'{sample_id}: {YZDExcpetion.TOO_FEW_IP_NODES}\n')
            raise YZDExcpetion(YZDExcpetion.TOO_FEW_IP_NODES, sample_id)

        selected_ip_indices_base = np.array(sorted(random.sample(range(num_ip),
                                                                 selected_num_ip)))
        # selected_ip_indices_base = np.array([2,3,4,5,57])
        # print(
        #     f'dfa_utils line 281, but for debug!!! selected_ip_base: {selected_ip_indices_base}; start_trace_idx = {start_trace_idx}')
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
                if task_name == 'dfa_liveness' or task_name == b'dfa_liveness':
                    trace_content_base[np.array(active_ip_list) - num_pp, np.repeat(pp, num_active_ip_node)] = 1
                else:
                    trace_content_base[np.array(active_ip_list), np.repeat(pp, num_active_ip_node)] = 1
            trace_content_sparse = trace_content_base[selected_ip_indices_base, :]
            # [selected_num_ip, num_pp]
            # trace_content_sparse = trace_content_sparse.transpose().reshape(-1, )  # [num_pp * selected_num_ip, ]
            trace_content_sparse = trace_content_sparse.reshape(-1, )  # [num_pp * selected_num_ip, ]
            if task_name == 'dfa_liveness' or task_name == b'dfa_liveness':

                trace_idx_source = np.repeat(np.arange(num_pp, num_pp + selected_num_ip),
                                             [num_pp] * selected_num_ip)
                # [num_pp * selected_num_ip, ]
            else:
                trace_idx_source = np.repeat(selected_ip_indices_base,
                                             [num_pp] * selected_num_ip)
            trace_idx_target = np.tile(np.arange(num_pp), selected_num_ip)
            # [num_pp * selected_num_ip, ]
            trace_sparse = np.concatenate([np.expand_dims(trace_idx_source, -1),
                                           np.expand_dims(trace_idx_target, -1),
                                           np.expand_dims(trace_content_sparse, -1)],
                                          axis=1)
            # print(f'dfa_utils line 321, cur_trace_idx = {cur_trace_idx - 1}, the shape is: {trace_sparse.shape} ')
            # print(trace_sparse)
            trace_list.append(trace_sparse)
        # print(f'length of trace_list = {len(trace_list)}')
        return trace_list, selected_ip_indices_base, num_pp

    def _load_sparse_edge_from_str(self, task_name: Union[str, bytes],
                                   edges_str,
                                   selected_ip_indices_base: np.ndarray):
        edges_saved_matrix = np.fromstring(edges_str, sep=' ', dtype=int)
        assert selected_ip_indices_base.ndim == 1 and selected_ip_indices_base.shape[0] == self.selected_num_ip
        if task_name == 'dfa_liveness' or task_name == b'dfa_liveness':
            edges_saved_matrix = edges_saved_matrix.reshape((-1, 3))
            # print(f'the shape of edges_saved_matrix is: {edges_saved_matrix.shape}')
            num_pp, num_ip = edges_saved_matrix[0, 0], edges_saved_matrix[0, 1]
            cfg_row_indices = np.where(edges_saved_matrix[:, -1] == 0)[0]
            gen_row_indices = np.where(edges_saved_matrix[:, -1] == 1)[0]
            kill_row_indices = np.where(edges_saved_matrix[:, -1] == 2)[0]
            cfg_edges_backward = edges_saved_matrix[cfg_row_indices, :-1]
            cfg_edges_forward = cfg_edges_backward[:, [1, 0]]
            gen_edges = edges_saved_matrix[gen_row_indices, :-1][:, [1, 0]]
            kill_edges = edges_saved_matrix[kill_row_indices, :-1][:, [1, 0]]

            gen_array_dense = np.zeros(shape=(num_ip, num_pp), dtype=int)
            kill_array_dense = np.zeros(shape=(num_ip, num_pp), dtype=int)
            gen_array_dense[gen_edges[:, 0] - num_pp, gen_edges[:, 1]] = 1
            kill_array_dense[kill_edges[:, 0] - num_pp, kill_edges[:, 1]] = 1

            gen_vectors = gen_array_dense[selected_ip_indices_base, :].transpose()
            # [num_pp, selected_num_ip, ]
            kill_vectors = kill_array_dense[selected_ip_indices_base, :].transpose()
            # [num_pp, selected_num_ip, ]
        else:
            edges_saved_matrix = edges_saved_matrix.reshape((-1, 2))
            num_pp = edges_saved_matrix[0, 0]
            if task_name == 'dfa_reachability' or task_name == b'dfa_reachability':
                cfg_edges_backward = edges_saved_matrix[1:, :]
                cfg_edges_forward = cfg_edges_backward[:, [1, 0]]
            else:
                assert task_name == 'dfa_dominance' or task_name == b'dfa_dominance'
                cfg_edges_forward = edges_saved_matrix[1:, :]
                cfg_edges_backward = cfg_edges_forward[:, [1, 0]]
            gen_array_dense = np.identity(num_pp, dtype=int)
            gen_vectors = gen_array_dense[selected_ip_indices_base, :].transpose()
            # [num_pp, selected_num_ip]
            kill_vectors = np.zeros(num_pp, self.selected_num_ip)
            # [num_pp, selected_num_ip]
        num_cfg_edges = cfg_edges_forward.shape[0]
        cfg_edges_forward = np.concatenate([cfg_edges_forward,
                                            np.ones((num_cfg_edges, 1))],
                                           axis=1)
        # [num_cfg, 3]
        cfg_edges_backward = np.concatenate([cfg_edges_backward,
                                             np.zeros((num_cfg_edges, 1))],
                                            axis=1)
        # [num_cfg, 3]
        cfg_edges = np.concatenate([cfg_edges_forward, cfg_edges_backward], axis=0)
        return [cfg_edges, gen_vectors, kill_vectors]


def _get_analyze_task_name(task_name: str):
    if task_name == 'dfa_liveness':
        return 'yzd_liveness'
    elif task_name == 'dfa_reachability':
        return 'yzd_reachability'
    elif task_name == 'dfa_dominance':
        return 'yzd_dominance'
    return task_name


def _get_activation(activation_str):
    if activation_str == 'relu':
        return jax.nn.relu
    raise YZDExcpetion(YZDExcpetion.UNRECOGNIZED_ACTIVATION_TYPE)


def parse_params(params_filepath: str):
    with open(params_filepath) as json_loader:
        params_dict = json.load(json_loader)

    train_sample_id_list = []
    with open(params_dict['dfa_sampler']['train_sample_id_savepath']) as train_sample_reader:
        for line in train_sample_reader.readlines():
            train_sample_id_list.append(line.strip())
    params_dict['dfa_sampler']['train_sample_id_list'] = train_sample_id_list
    del params_dict['dfa_sampler']['train_sample_id_savepath']
    vali_sample_id_list = []
    with open(params_dict['dfa_sampler']['vali_sample_id_savepath']) as vali_sample_reader:
        for line in vali_sample_reader.readlines():
            vali_sample_id_list.append(line.strip())
    params_dict['dfa_sampler']['vali_sample_id_list'] = vali_sample_id_list
    del params_dict['dfa_sampler']['vali_sample_id_savepath']
    test_sample_id_list = []
    with open(params_dict['dfa_sampler']['test_sample_id_savepath']) as test_sample_reader:
        for line in test_sample_reader.readlines():
            test_sample_id_list.append(line.strip())
    params_dict['dfa_sampler']['test_sample_id_list'] = test_sample_id_list
    del params_dict['dfa_sampler']['test_sample_id_savepath']

    params_dict['dfa_net']['spec'] = [dfa_specs.DFASPECS[params_dict['task']['task_name']]]
    params_dict['processor']['activation'] = _get_activation(params_dict['processor']['activation_name'])
    del params_dict['processor']['activation_name']
    return params_dict


def dim_expand_to(x, y):
    while len(y.shape) > len(x.shape):
        x = jnp.expand_dims(x, -1)
    return x
