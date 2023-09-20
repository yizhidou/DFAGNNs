from clrs._src import dfa_sampler
from clrs._src import yzd_utils

def FeaturesListGenerator():
    pass

if __name__ == '__main__':
    sample_path_processor = yzd_utils.SamplePathProcessor(sourcegraph_dir='',
                                                          errorlog_savepath='',
                                                          dataset_savedir='',
                                                          statistics_savepath='')
    sample_loader = yzd_utils.SampleLoader(sample_path_processor=sample_path_processor,
                                           max_iteration=0,
                                           max_num_pp=0,
                                           gkt_edges_rate=0,
                                           selected_num_ip=0)