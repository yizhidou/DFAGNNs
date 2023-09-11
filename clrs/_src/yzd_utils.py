from typing import List, Union, Dict
import os, json
import numpy as np
from programl.proto import *
import programl
import random
from clrs._src import yzd_probing

_ArraySparse = yzd_probing.ArraySparse
_ArrayDense = yzd_probing.ArrayDense
_Array = yzd_probing.Array
taskname_shorts = dict(yzd_liveness='yl', yzd_dominance='yd', yzd_reachability='yr')


class YZDExcpetion(Exception):
    # the current sample has been previously recognized as errored
    RECORDED_ERRORED_SAMPLE = 1
    # newly recognized error sample
    NEWLY_ERRORED_SAMPLE = 2
    # too few interested points error (less than selected num)
    TOO_FEW_IP_NODES = 3
    # too much program points error
    TOO_MANY_PP_NODES = 4

    def __init__(self, error_code: int,
                 sample_id: Union[str, None] = None):
        self.error_code = error_code
        self.sample_id = sample_id
        super().__init__()

    def error_msg(self):
        if self.error_code == self.RECORDED_ERRORED_SAMPLE:
            msg = 'This sample has previously been recorded as errored!'
        elif self.error_code == self.NEWLY_ERRORED_SAMPLE:
            msg = 'This sample is newly recognized errored, we have now recorded it!'
        else:
            msg = 'unrecognized error!'
        if self.sample_id:
            msg += f' sample_id: {self.sample_id}'
        return msg


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
                 errorlog_savepath: str,
                 dataset_savedir: Union[None, str] = None,
                 statistics_savepath: Union[None, str] = None):
        if not os.path.isdir(sourcegraph_dir):
            # YZDTODO raise an error
            pass
        self.sourcegraph_dir = sourcegraph_dir
        self.errored_sample_ids = {}
        if not os.path.isfile(errorlog_savepath):
            os.system(f'touch {errorlog_savepath}')
        else:
            with open(errorlog_savepath) as errored_reader:
                for line in errored_reader.readlines():
                    self.errored_sample_ids[line.strip()] = 1
        self.errorlog_savepath = errorlog_savepath
        self.dataset_savedir = dataset_savedir
        self.statistics_savepath = statistics_savepath

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
                 max_num_pp: int,
                 gkt_edges_rate: int,
                 if_sync: bool = False,
                 if_idx_reorganized: bool = True,
                 if_save: bool = False,
                 if_sparse: bool = True,
                 selected_num_ip: int = 5):
        self.sample_path_processor = sample_path_processor
        self.max_iteration = max_iteration
        self.expected_hint_len = self.max_iteration - 1
        self.gkt_edges_rate = gkt_edges_rate
        self.max_num_pp = max_num_pp
        self.if_sync = if_sync
        self.if_idx_reorganized = if_idx_reorganized
        self.if_save = if_save
        self.if_sparse = if_sparse
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
        # self.rng = np.random.default_rng()

    def load_a_sample(self, task_name, sample_id):
        if sample_id in self.sample_path_processor.errored_sample_ids:
            raise YZDExcpetion(YZDExcpetion.RECORDED_ERRORED_SAMPLE, sample_id)
        if self.sample_path_processor.if_sample_exists(task_name=task_name, sample_id=sample_id):
            trace_savepath = self.sample_path_processor.trace_savepath(task_name, sample_id)
            with open(trace_savepath, 'rb') as result_reader:
                trace_bytes_from_file = result_reader.read()
            edge_savepath = self.sample_path_processor.edge_savepath(task_name, sample_id)
            with open(edge_savepath) as edges_reader:
                edges_bytes_from_file = edges_reader.read()
            if self.if_sparse:
                trace_list, selected_ip_indices_base, num_pp = self._load_sparse_trace_from_bytes(task_name=task_name,
                                                                                                  trace_bytes=trace_bytes_from_file,
                                                                                                  sample_id=sample_id,
                                                                                                  selected_num_ip=self.selected_num_ip)
                array_list = self._load_sparse_edge_from_str(task_name=task_name,
                                                             edges_str=edges_bytes_from_file,
                                                             selected_ip_indices_base=selected_ip_indices_base)
                if_pp, if_ip = self._get_node_type(task_name=task_name,
                                                   selected_ip_indices_base=selected_ip_indices_base,
                                                   num_pp=num_pp)
                return trace_list, array_list, if_pp, if_ip
            else:
                trace_list = self._load_dense_trace_from_bytes(task_name=task_name,
                                                               trace_bytes=trace_bytes_from_file)
                array_list = self._load_dense_edge_from_str(task_name=task_name,
                                                            edges_str=edges_bytes_from_file)
                return trace_list, array_list
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
            if len(cpperror) > 0:
                self.sample_path_processor.errored_sample_ids[sample_id] = 1
                with open(self.sample_path_processor.errorlog_savepath, 'a') as error_sample_writer:
                    error_sample_writer.write(sample_id + '\n')
                raise YZDExcpetion(YZDExcpetion.NEWLY_ERRORED_SAMPLE, sample_id)
            sample_statistics, edge_chunck, trace_chunck = self._parse_cpp_stdout(cpp_out)
            if self.sample_path_processor.statistics_savepath:
                self._merge_statistics(sample_id, sample_statistics)
            if self.if_sparse:
                trace_list, selected_ip_indices_base, num_pp = self._load_sparse_trace_from_bytes(task_name=task_name,
                                                                                                  trace_bytes=trace_chunck,
                                                                                                  sample_id=sample_id,
                                                                                                  selected_num_ip=self.selected_num_ip)
                array_list = self._load_sparse_edge_from_str(task_name=task_name,
                                                             edges_str=edge_chunck,
                                                             selected_ip_indices_base=selected_ip_indices_base)
                if_pp, if_ip = self._get_node_type(task_name=task_name,
                                                   selected_ip_indices_base=selected_ip_indices_base,
                                                   num_pp=num_pp)
                return trace_list, array_list, if_pp, if_ip
            else:
                trace_list = self._load_dense_trace_from_bytes(task_name, trace_chunck)
                array_list = self._load_dense_edge_from_str(task_name, edge_chunck)
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
        # print(f'line be is: {byte_str_be}; end_idx = {end_idx_be}')
        end_idx_pp, byte_str_pp = _get_a_line(star_idx=end_idx_be)
        # print(f'line pp is: {byte_str_pp}; end_idx = {end_idx_pp}')
        _, _, num_pp = _parse_a_line(byte_str_pp)
        sample_statistics['num_pp'] = num_pp
        end_idx_ip, byte_str_ip = _get_a_line(star_idx=end_idx_pp)
        # print(f'line ip is: {byte_str_ip}; end_idx = {end_idx_ip}')
        _, _, num_ip = _parse_a_line(byte_str_ip)
        sample_statistics['num_ip'] = num_ip
        end_idx_it, byte_str_it = _get_a_line(star_idx=end_idx_ip)
        # print(f'line it is: {byte_str_it}; end_idx = {end_idx_it}')
        _, item_name, num_iteration = _parse_a_line(byte_str_it)
        sample_statistics[f'{item_name}_{task_name}'] = num_iteration
        end_idx_du, byte_str_du = _get_a_line(star_idx=end_idx_it)
        # print(f'line be du: {byte_str_du}; end_idx = {end_idx_du}')
        end_idx_edge_size, byte_str_edge_size = _get_a_line(star_idx=end_idx_du)
        # print(f'line size is: {byte_str_edge_size}; end_idx = {end_idx_edge_size}')
        task_name_in_byte, _, edge_size = _parse_a_line(byte_str_edge_size)
        edge_chunck = cpp_out[end_idx_edge_size + 1: end_idx_edge_size + 1 + edge_size]
        trace_chunck = cpp_out[end_idx_edge_size + 1 + edge_size:]
        return sample_statistics, edge_chunck, trace_chunck

    def _load_dense_trace_from_bytes(self, task_name: Union[str, bytes], trace_bytes: bytes):
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
                # target_node_list = list(trace_of_this_iteration[source_node].value)
                target_node_list = trace_of_this_iteration[source_node].value
                num_target_node = len(target_node_list)
                if num_target_node == 0:
                    continue
                # 相应值填进trace_matrix里
                trace_matrix[np.repeat(source_node, num_target_node), np.array(target_node_list)] = 1
            trace_list.append(trace_matrix)
        return trace_list

    def _load_dense_edge_from_str(self, task_name: Union[str, bytes], edges_str):
        edges_saved_matrix = np.fromstring(edges_str, sep=' ', dtype=int)
        if task_name == 'yzd_liveness' or task_name == b'yzd_liveness':
            edges_saved_matrix = edges_saved_matrix.reshape((-1, 3))
            # print(f'the shape of edges_saved_matrix is: {edges_saved_matrix.shape}')
            num_pp, num_ip = edges_saved_matrix[0, 0], edges_saved_matrix[0, 1]
            num_node = num_pp + num_ip
            print(f'num_pp = {num_pp}; num_ip = {num_ip}; total = {num_pp + num_ip}')
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

    def _load_sparse_trace_from_bytes(self, task_name: Union[str, bytes],
                                      trace_bytes: bytes,
                                      sample_id: str,
                                      selected_num_ip: int):
        result_obj = ResultsEveryIteration()
        result_obj.ParseFromString(trace_bytes)
        num_pp = len(result_obj.program_points.value)
        if num_pp > self.max_num_pp:
            self.sample_path_processor.errored_sample_ids[sample_id] = 1
            with open(self.sample_path_processor.errorlog_savepath, 'a') as error_sample_writer:
                error_sample_writer.write(sample_id + '\n')
            raise YZDExcpetion(YZDExcpetion.TOO_MANY_PP_NODES, sample_id)
        num_ip = len(result_obj.interested_points.value)
        if num_ip < selected_num_ip:
            self.sample_path_processor.errored_sample_ids[sample_id] = 1
            with open(self.sample_path_processor.errorlog_savepath, 'a') as error_sample_writer:
                error_sample_writer.write(sample_id + '\n')
            raise YZDExcpetion(YZDExcpetion.TOO_FEW_IP_NODES, sample_id)

        selected_ip_indices_base = np.array(sorted(random.sample(range(num_ip),
                                                                 selected_num_ip)))
        if not (task_name == 'yzd_liveness' or task_name == b'yzd_liveness'):
            assert num_pp == num_ip
        # if task_name == 'yzd_liveness' or task_name == b'yzd_liveness':
        # num_node = num_pp + num_ip
        # selected_ip_indices = selected_ip_indices_base + num_pp
        # else:
        #     assert num_pp == num_ip
        # num_node = num_pp
        # selected_ip_indices = selected_ip_indices_base
        trace_len = len(result_obj.results_every_iteration)  # this should be num_iteration+1
        trace_list = []
        for trace_idx in range(trace_len):
            trace_content_base = np.zeros(shape=(num_pp, num_ip))
            trace_of_this_iteration = result_obj.results_every_iteration[trace_idx].result_map
            # 要从这一个trace里转化出一个matrix来
            assert len(trace_of_this_iteration) == num_pp
            for source_node in trace_of_this_iteration.keys():
                # target_node_list = list(trace_of_this_iteration[source_node].value)
                target_node_list = trace_of_this_iteration[source_node].value
                num_target_node = len(target_node_list)
                if num_target_node == 0:
                    continue
                # 相应值填进trace_matrix里
                trace_content_base[np.repeat(source_node, num_target_node), np.array(target_node_list) - num_pp] = 1
            trace_content_sparse = trace_content_base[np.arange(num_pp), selected_ip_indices_base]
            # [num_pp, selected_num_ip]
            trace_content_sparse = trace_content_sparse.transpose().reshape(-1, )  # [num_pp * selected_num_ip, ]
            if task_name == 'yzd_liveness' or task_name == b'yzd_liveness':

                trace_idx_col_sparse = np.repeat(np.arange(num_pp, num_pp + selected_num_ip),
                                                 [num_pp] * selected_num_ip)
                # [num_pp * selected_num_ip, ]
            else:
                trace_idx_col_sparse = np.repeat(selected_ip_indices_base,
                                                 [num_pp] * selected_num_ip)
            trace_idx_row_sparse = np.tile(np.arange(num_pp), selected_num_ip)
            # [num_pp * selected_num_ip, ]
            trace_sparse_data = np.concatenate([np.expand_dims(trace_idx_row_sparse, -1),
                                                np.expand_dims(trace_idx_col_sparse, -1),
                                                np.expand_dims(trace_content_sparse, -1)],
                                               axis=1)
            trace_sparse = _ArraySparse(edges_with_optional_content=trace_sparse_data,
                                        nb_nodes=(num_pp + selected_num_ip) if (
                                                task_name == 'yzd_liveness' or
                                                task_name == b'yzd_liveness') else num_pp,
                                        nb_edges=num_pp * selected_num_ip)

            trace_list.append(trace_sparse)
        return trace_list, selected_ip_indices_base, num_pp

    def _get_node_type(self, task_name: Union[str, bytes],
                       selected_ip_indices_base,
                       num_pp):
        assert selected_ip_indices_base.ndim == 1 and selected_ip_indices_base.shape[0] == self.selected_num_ip
        if task_name == 'yzd_liveness' or task_name == b'yzd_liveness':
            if_pp = np.concatenate([np.ones(shape=(num_pp,), dtype=int),
                                    np.zeros(shape=(self.selected_num_ip,), dtype=int)],
                                   axis=0)
            if_ip = np.concatenate([np.zeros(shape=(num_pp,), dtype=int),
                                    np.ones(shape=(self.selected_num_ip,), dtype=int)],
                                   axis=0)
        else:
            if_pp = np.ones(shape=(num_pp,), dtype=int)
            if_ip = np.zeros_like(if_pp)
            if_ip[selected_ip_indices_base] = 1
        return if_pp, if_ip

    def _load_sparse_edge_from_str(self, task_name: Union[str, bytes],
                                   edges_str,
                                   selected_ip_indices_base: np.ndarray):
        edges_saved_matrix = np.fromstring(edges_str, sep=' ', dtype=int)
        assert selected_ip_indices_base.ndim == 1 and selected_ip_indices_base.shape[0] == self.selected_num_ip
        if task_name == 'yzd_liveness' or task_name == b'yzd_liveness':
            edges_saved_matrix = edges_saved_matrix.reshape((-1, 3))
            # print(f'the shape of edges_saved_matrix is: {edges_saved_matrix.shape}')
            num_pp, num_ip = edges_saved_matrix[0, 0], edges_saved_matrix[0, 1]
            # num_node = num_pp + num_ip
            print(f'num_pp = {num_pp}; num_ip = {num_ip}; total = {num_pp + num_ip}')
            cfg_row_indices = np.where(edges_saved_matrix[:, -1] == 0)[0]
            gen_row_indices = np.where(edges_saved_matrix[:, -1] == 1)[0]
            kill_row_indices = np.where(edges_saved_matrix[:, -1] == 2)[0]
            cfg_sparse = edges_saved_matrix[cfg_row_indices, :-1]
            gen_edges = edges_saved_matrix[gen_row_indices, :-1]
            kill_edges = edges_saved_matrix[kill_row_indices, :-1]

            # cfg_array = np.zeros(shape=(num_node, num_node), dtype=int)
            gen_array_dense = np.zeros(shape=(num_pp, num_ip), dtype=int)
            kill_array_dense = np.zeros(shape=(num_pp, num_ip), dtype=int)
            # print(f'the shape of cfg_array is: {cfg_array.shape}; the shape of cfg_edges is: {cfg_edges.shape}')
            # cfg_array[cfg_edges[:, 0], cfg_edges[:, 1]] = 1
            gen_array_dense[gen_edges[:, 0], gen_edges[:, 1] - num_pp] = 1
            kill_array_dense[kill_edges[:, 0], kill_edges[:, 1] - num_pp] = 1

            gen_content_sparse = gen_array_dense[:, selected_ip_indices_base].transpose().reshape(-1, )
            # [num_pp * selected_num_ip, ]
            kill_content_sparse = kill_array_dense[:, selected_ip_indices_base].transpose().reshape(-1, )
            # [num_pp * selected_num_ip, ]
            gen_kill_idx_row_sparse = np.tile(np.arange(num_pp), self.selected_num_ip)
            # [num_pp * selected_num_ip, ]
            gen_kill_idx_col_sparse = np.repeat(np.arange(num_pp, num_pp + self.selected_num_ip),
                                                [num_pp] * self.selected_num_ip)
            # [num_pp * selected_num_ip, ]
            gen_sparse = np.concatenate([np.expand_dims(gen_kill_idx_row_sparse, -1),
                                         np.expand_dims(gen_kill_idx_col_sparse, -1),
                                         np.expand_dims(gen_content_sparse, -1)],
                                        axis=1)
            kill_sparse = np.concatenate([np.expand_dims(gen_kill_idx_row_sparse, -1),
                                          np.expand_dims(gen_kill_idx_col_sparse, -1),
                                          np.expand_dims(kill_content_sparse, -1)],
                                         axis=1)

            cfg_sparse_array = _ArraySparse(edges_with_optional_content=cfg_sparse,
                                            nb_nodes=num_pp + self.selected_num_ip,
                                            nb_edges=cfg_sparse.shape[0])
            gen_sparse_array = _ArraySparse(edges_with_optional_content=gen_sparse,
                                            nb_nodes=num_pp + self.selected_num_ip,
                                            nb_edges=num_pp * self.selected_num_ip)
            kill_sparse_array = _ArraySparse(edges_with_optional_content=kill_sparse,
                                             nb_nodes=num_pp + self.selected_num_ip,
                                             nb_edges=num_pp * self.selected_num_ip)
            return [cfg_sparse_array, gen_sparse_array, kill_sparse_array]
        else:
            edges_saved_matrix = edges_saved_matrix.reshape((-1, 2))
            num_pp = edges_saved_matrix[0, 0]
            cfg_sparse = edges_saved_matrix[1:, :]
            # cfg_source = edges_saved_matrix[1:, 0]
            # cfg_target = edges_saved_matrix[1:, 1]
            # cfg_array = np.zeros(shape=(num_node, num_node), dtype=int)
            # cfg_array[cfg_source, cfg_target] = 1
            gen_array_dense = np.identity(num_pp, dtype=int)
            gen_content_sparse = gen_array_dense[:, selected_ip_indices_base].transpose().reshape(-1, )
            # [num_pp * selected_num_ip, ]
            gen_idx_row_sparse = np.tile(np.arange(num_pp), self.selected_num_ip)
            # [num_pp * selected_num_ip, ]
            gen_idx_col_sparse = np.repeat(selected_ip_indices_base,
                                           [num_pp] * self.selected_num_ip)
            gen_sparse = np.concatenate([np.expand_dims(gen_idx_row_sparse, -1),
                                         np.expand_dims(gen_idx_col_sparse, -1),
                                         np.expand_dims(gen_content_sparse, -1)],
                                        axis=1)

            cfg_sparse_array = _ArraySparse(edges_with_optional_content=cfg_sparse,
                                            nb_nodes=num_pp,
                                            nb_edges=cfg_sparse.shape[0])
            gen_sparse_array = _ArraySparse(edges_with_optional_content=gen_sparse,
                                            nb_nodes=num_pp,
                                            nb_edges=num_pp * self.selected_num_ip)
            return [cfg_sparse_array, gen_sparse_array]
