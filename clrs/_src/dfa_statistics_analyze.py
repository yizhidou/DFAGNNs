import json, argparse
from clrs._src import dfa_utils


def count_filtered_samples(sample_id_savepath: str,
                           full_statistics_filepath: str,
                           errored_sample_filepath: str,
                           max_num_pp: int,
                           min_num_pp: int,
                           cfg_edges_rate: int):
    with open(errored_sample_filepath) as errored_sample_ids_loader:
        errored_sample_ids = json.load(errored_sample_ids_loader)
    filtered_samples = dfa_utils.filter_sample_list(full_statistics_savepath=full_statistics_filepath,
                                                    errored_sample_ids=errored_sample_ids,
                                                    max_num_pp=max_num_pp,
                                                    min_num_pp=min_num_pp,
                                                    cfg_edges_rate=cfg_edges_rate,
                                                    sample_id_savepath=sample_id_savepath)
    print(f'there are {len(filtered_samples)} in total!')


if __name__ == '__main__':
    full_statistics_filepath = "/data_hdd/lx20/yzd_workspace/Datasets/Statistics/poj104_Statistics/poj104_full_statistics.json"
    errored_sample_filepath = "/data_hdd/lx20/yzd_workspace/Logs/ErrorLogs/poj104_errors_max500.txt"
    parser = argparse.ArgumentParser(description='Please input the params filename')
    parser.add_argument('--sample_ids', type=str, required=True)
    parser.add_argument('--max_num_pp', type=int, required=True)
    parser.add_argument('--min_num_pp', type=int, required=True)
    parser.add_argument('--cfg_edges_rate', type=float, default=1.5)
    args = parser.parse_args()
    count_filtered_samples(sample_id_savepath=args.sample_ids,
                           full_statistics_filepath=full_statistics_filepath,
                           errored_sample_filepath=errored_sample_filepath,
                           max_num_pp=args.max_num_pp,
                           min_num_pp=args.min_num_pp,
                           cfg_edges_rate=args.cfg_edges_rate)
