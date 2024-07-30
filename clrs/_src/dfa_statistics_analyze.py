import json, argparse
from clrs._src import dfa_utils
from typing import Union, List


def count_filtered_samples(full_statistics_filepath: str,
                           errored_sample_filepath: str,
                           max_num_pp: int,
                           min_num_pp: int,
                           cfg_edges_rate: int,
                           sample_ids: Union[None, List]=None):
    with open(errored_sample_filepath) as errored_sample_ids_loader:
        errored_sample_ids = json.load(errored_sample_ids_loader)
    with open(full_statistics_filepath) as statistics_loader:
        full_statistics_dict = json.load(statistics_loader)
    candi_sample_ids = full_statistics_dict.keys()
    # def full_statistics_dict
    filtered_samples = dfa_utils.filter_sample_list(full_statistics_savepath=full_statistics_filepath,
                                                    errored_sample_ids=errored_sample_ids,
                                                    max_num_pp=max_num_pp,
                                                    min_num_pp=min_num_pp,
                                                    cfg_edges_rate=cfg_edges_rate,
                                                    sample_ids=candi_sample_ids if sample_ids is None else sample_ids)
    min_trace_len = 999999 
    max_trace_len = 0
    for sample in filtered_samples:
        trace_len_1, trace_len_2, trace_len_3 = full_statistics_dict[sample][-3:] 
        min_trace_len = min(min_trace_len, trace_len_1, trace_len_2, trace_len_3)
        max_trace_len = max(max_trace_len, trace_len_1, trace_len_2, trace_len_3)
    print(f'there are {len(filtered_samples)} in total! min_trace_len = {min_trace_len}; max_trace_len = {max_trace_len}')


if __name__ == '__main__':
    # full_statistics_filepath = "/data_hdd/lx20/yzd_workspace/Datasets/Statistics/poj104_Statistics/poj104_full_statistics.json"
    # errored_sample_filepath = "/data_hdd/lx20/yzd_workspace/Logs/ErrorLogs/poj104_errors_max500.txt"
    parser = argparse.ArgumentParser(description='Please input the params filename')
    parser.add_argument('--sample_id_savepath', type=str, default=None, required=False)
    parser.add_argument('--full_statistics_filepath', type=str, required=True)
    parser.add_argument('--errored_sample_filepath', type=str, required=True)
    parser.add_argument('--max_num_pp', type=int, required=True)
    parser.add_argument('--min_num_pp', type=int, required=True)
    parser.add_argument('--cfg_edges_rate', type=float, default=1.5)
    args = parser.parse_args()
    if args.sample_id_savepath is not None:
        candi_sample_ids = []
        with open(args.sample_id_savepath) as sample_id_loader:
            for line in sample_id_loader.readlines():
                sample_id = line.strip()
                candi_sample_ids.append(sample_id)
    else:
        candi_sample_ids = None
    count_filtered_samples(sample_ids=candi_sample_ids,
                           full_statistics_filepath=args.full_statistics_filepath,
                           errored_sample_filepath=args.errored_sample_filepath,
                           max_num_pp=args.max_num_pp,
                           min_num_pp=args.min_num_pp,
                           cfg_edges_rate=args.cfg_edges_rate)
