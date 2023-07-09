from typing import List, Union
import os
import numpy as np
from programl.proto import *
import programl

_Array = np.ndarray
taskname_shorts = dict(yzd_liveness='yl', yzd_dominance='yd', yzd_reachability='yr')


class ParamsParser:
    def __init__(self, params_savepath):
        if not os.path.isfile(params_savepath):
            pass
            # YZDTODO raise an error
        self.params_savepath = params_savepath
        self.params_dict = self._parse_params()
        self.max_iteration = 0

    def _parse_params(self):
        params_dict = dict()
        with open(self.params_savepath) as params_reader:
            for line in params_reader.readlines():
                param_name, param_value = line.split(': ')
                param_name = param_name.strip()
                if param_name == 'sourcegraph_dir':
                    params_dict['sourcegraph_dir'] = param_value.strip()
                if param_name == 'dataset_savedir':
                    params_dict['dataset_savedir'] = param_value.strip()
        return params_dict

    @property
    def sourcegraph_dir(self):
        return self.params_dict['sourcegraph_dir']

    @property
    def dataset_savedir(self):
        return self.params_dict['dataset_savedir']


class SamplePathProcessor:
    def __init__(self, sourcegraph_dir: str,
                 dataset_savedir: Union[None, str] = None):
        if not os.path.isdir(sourcegraph_dir):
            # YZDTODO raise an error
            pass
        self.sourcegraph_dir = sourcegraph_dir
        self.dataset_savedir = dataset_savedir

    def sourcegraph_savepath(self, sample_id):
        # YZDTODO  这里没有写完
        print(f'sample_id = {sample_id}')
        return os.path.join(self.sourcegraph_dir, sample_id + '.ProgramGraph.pb')

    def _trace_savedir(self, task_name):
        return os.path.join(self.dataset_savedir, task_name, 'Traces')

    def _edge_savedir(self, task_name):
        return os.path.join(self.dataset_savedir, task_name, 'Edges')

    def trace_savepath(self, task_name, sample_id):
        return os.path.join(self._trace_savedir(task_name), sample_id + f'.{taskname_shorts[task_name]}.trace')

    def edge_savepath(self, task_name, sample_id):
        return os.path.join(self._edge_savedir(task_name), sample_id + f'.{taskname_shorts[task_name]}.edge')

    def if_sample_exists(self, task_name, sample_id):
        if not self.dataset_savedir:
            return False
        trace_path_to_check = self.trace_savepath(task_name, sample_id)
        edge_path_to_check = self.edge_savepath(task_name, sample_id)
        if not os.path.isfile(trace_path_to_check) or not os.path.isfile(edge_path_to_check):
            return False
        if os.path.getsize(trace_path_to_check) == 0 or os.path.getsize(edge_path_to_check) == 0:
            return False
        return True


class SampleLoader:
    def __init__(self, sample_path_processor: SamplePathProcessor,
                 max_iteration: int,
                 if_sync: bool = False,
                 if_idx_reorganized: bool = True,
                 if_save=False
                 ):
        self.sample_path_processor = sample_path_processor
        self.max_iteration = max_iteration
        self.if_sync = if_sync
        self.if_idx_reorganized = if_idx_reorganized
        self.if_save = if_save
        if not self.sample_path_processor.dataset_savedir:
            self.if_save = False

    def load_a_sample(self, task_name, sample_id):
        if self.sample_path_processor.if_sample_exists(task_name=task_name, sample_id=sample_id):
            trace_savepath = self.sample_path_processor.trace_savepath(task_name, sample_id)
            with open(trace_savepath, 'rb') as result_reader:
                trace_bytes_from_file = result_reader.read()
            trace_list = self._load_trace_from_bytes(task_name, trace_bytes_from_file)
            edge_savepath = self.sample_path_processor.edge_savepath(task_name, sample_id)
            with open(edge_savepath) as edges_reader:
                edges_bytes_from_file = edges_reader.read()
            array_list = self._load_edge_from_str(task_name, edges_bytes_from_file)
        else:
            cpp_out, cpperror = programl.yzd_analyze(task_name=task_name,
                                                     max_iteration=self.max_iteration,
                                                     program_graph_sourcepath=self.sample_path_processor.sourcegraph_savepath(
                                                         sample_id),
                                                     edge_list_savepath=self.sample_path_processor.edge_savepath(
                                                         task_name, sample_id) if self.if_save else None,
                                                     result_savepath=self.sample_path_processor.trace_savepath(
                                                         task_name, sample_id) if self.if_save else None,
                                                     if_sync=self.if_sync,
                                                     if_idx_reorganized=self.if_idx_reorganized)
            #     YZDTODO  这里应该加错误logging
            assert len(cpperror) == 0
            edge_chunck, trace_chunck = self._parse_cpp_stdout(cpp_out)
            trace_list = self._load_trace_from_bytes(task_name, trace_chunck)
            array_list = self._load_edge_from_str(task_name, edge_chunck)
        return trace_list, array_list

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
            return task_name_in_byte, item_name_in_byte, int(num.decode())

        # sample_statistics = dict()
        end_idx_be, byte_str_be = _get_a_line(star_idx=0)
        # print(f'line be is: {byte_str_be}; end_idx = {end_idx_be}')
        end_idx_pp, byte_str_pp = _get_a_line(star_idx=end_idx_be)
        # print(f'line pp is: {byte_str_pp}; end_idx = {end_idx_pp}')
        end_idx_ip, byte_str_ip = _get_a_line(star_idx=end_idx_pp)
        # print(f'line ip is: {byte_str_ip}; end_idx = {end_idx_ip}')
        end_idx_it, byte_str_it = _get_a_line(star_idx=end_idx_ip)
        # print(f'line it is: {byte_str_it}; end_idx = {end_idx_it}')
        end_idx_du, byte_str_du = _get_a_line(star_idx=end_idx_it)
        # print(f'line be du: {byte_str_du}; end_idx = {end_idx_du}')
        end_idx_edge_size, byte_str_edge_size = _get_a_line(star_idx=end_idx_du)
        # print(f'line size is: {byte_str_edge_size}; end_idx = {end_idx_edge_size}')
        task_name_in_byte, _, edge_size = _parse_a_line(byte_str_edge_size)
        edge_chunck = cpp_out[end_idx_edge_size + 1: end_idx_edge_size + 1 + edge_size]
        trace_chunck = cpp_out[end_idx_edge_size + 1 + edge_size:]
        return edge_chunck, trace_chunck

    def _load_trace_from_bytes(self, task_name: Union[str, bytes], trace_bytes: bytes):
        result_obj = ResultsEveryIteration()
        result_obj.ParseFromString(trace_bytes)
        num_pp = len(result_obj.program_points.value)
        num_ip = len(result_obj.interested_points.value)
        if task_name == 'yzd_liveness' or task_name == b'yzd_liveness':
            num_node = num_pp + num_ip
        else:
            assert num_pp == num_ip
            num_node = num_pp
        trace_len = len(result_obj.results_every_iteration)  # this should be num_iteration+1
        trace_list = []
        for trace_idx in range(trace_len):
            trace_matrix = np.zeros(shape=(num_node, num_node))
            trace_of_this_iteration = result_obj.results_every_iteration[trace_idx].result_map
            # 要从这一个trace里转化出一个matrix来
            assert len(trace_of_this_iteration) == num_pp
            for source_node in trace_of_this_iteration.keys():
                target_node_list = list(trace_of_this_iteration[source_node].value)
                num_target_node = len(target_node_list)
                if num_target_node == 0:
                    continue
                # 相应值填进trace_matrix里
                trace_matrix[np.repeat(source_node, num_target_node), np.array(target_node_list)] = 1
            trace_list.append(trace_matrix)
        return trace_list

    def _load_edge_from_str(self, task_name: Union[str, bytes], edges_str):
        edges_saved_matrix = np.fromstring(edges_str, sep=' ', dtype=int)
        if task_name == 'yzd_liveness' or task_name == b'yzd_liveness':
            edges_saved_matrix = edges_saved_matrix.reshape((-1, 3))
            # print(f'the shape of edges_saved_matrix is: {edges_saved_matrix.shape}')
            num_pp, num_ip = edges_saved_matrix[0, 0], edges_saved_matrix[0, 1]
            num_node = num_pp + num_ip
            print(f'num_pp = {num_pp}; num_ip = {num_ip}')
            cfg_row_indices = np.where(edges_saved_matrix[:, -1] == 0)[0]
            gen_row_indices = np.where(edges_saved_matrix[:, -1] == 1)[0]
            kill_row_indices = np.where(edges_saved_matrix[:, -1] == 2)[0]
            cfg_edges = edges_saved_matrix[cfg_row_indices, :-1]
            gen_edges = edges_saved_matrix[gen_row_indices, :-1]
            kill_edges = edges_saved_matrix[kill_row_indices, :-1]
            cfg_array = np.zeros(shape=(num_node, num_node), dtype=int)
            gen_array = np.zeros(shape=(num_node, num_node), dtype=int)
            kill_array = np.zeros(shape=(num_node, num_node), dtype=int)
            # print(f'the shape of cfg_array is: {cfg_array.shape}; the shape of cfg_edges is: {cfg_edges.shape}')
            cfg_array[cfg_edges[:, 0], cfg_edges[:, 1]] = 1
            gen_array[gen_edges[:, 0], gen_edges[:, 1]] = 1
            kill_array[kill_edges[:, 0], kill_edges[:, 1]] = 1
            return [cfg_array, gen_array, kill_array]
        else:
            edges_saved_matrix = edges_saved_matrix.reshape((-1, 2))
            num_node = edges_saved_matrix[0, 0]
            cfg_source = edges_saved_matrix[1:, 0]
            cfg_target = edges_saved_matrix[1:, 1]
            cfg_array = np.zeros(shape=(num_node, num_node), dtype=int)
            cfg_array[cfg_source, cfg_target] = 1
            kill_array = np.identity(num_node)
            return [cfg_array, kill_array]
