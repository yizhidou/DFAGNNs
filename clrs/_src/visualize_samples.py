import argparse
import os

from clrs._src import dfa_utils

def _get_dataset_name(sample_id: str):
    dataset_name = None
    if sample_id.startswith('poj104'):
        dataset_name = 'poj104'
    elif sample_id.startswith('tensorflow'):
        dataset_name = 'tensorflow'
    elif sample_id.startswith('opencv'):
        dataset_name = 'opencv'
    elif sample_id.startswith('opencl'):
        dataset_name = 'opencl'
    elif sample_id.startswith('github'):
        dataset_name = 'github'
    else:
        print('Unrecognized dataset_name from sample_id! Please check!')
        exit(6)
    return dataset_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Please input the params filename')
    parser.add_argument('--sample_ids', type=str, nargs="+", default=None, required=False)
    args = parser.parse_args()
    dataset_name = _get_dataset_name(args.sample_ids[0])
    sourcegraph_dir = '/data_hdd/lx20/yzd_workspace/Datasets/ProgramlDatasetUnzip/dataflow/graphs'
    errorlog_savepath = f'/data_hdd/lx20/yzd_workspace/Logs/ErrorLogs/{dataset_name}_errors_max500.txt'
    visualized_sample_savedir = f'/data_hdd/lx20/yzd_workspace/Datasets/VisualizedSamples/{dataset_name}Visualized/'
    sample_path_processor = dfa_utils.SamplePathProcessor(
        sourcegraph_dir=sourcegraph_dir,
        errorlog_savepath=errorlog_savepath)
    sample_loader = dfa_utils.SampleLoader(sample_path_processor=sample_path_processor,
                                           expected_trace_len=6,
                                           max_num_pp=None,
                                           min_num_pp=None,
                                           cfg_edges_rate=1.5,
                                           selected_num_ip=5,
                                           dfa_version=0,
                                           if_sync=True,
                                           trace_sample_from_start=True,
                                           seed=6)
    for sample_id in args.sample_ids:
        assert dataset_name == _get_dataset_name(sample_id)
        print(f'{sample_id} is on processing...')
        visualized_sample_savepath = os.path.join(visualized_sample_savedir, f'{sample_id}.pdf')
        sample_loader.visualize_the_sample(sample_id=sample_id,
                                           savepath=visualized_sample_savepath)
        print('done!')
