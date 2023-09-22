import haiku as hk
import jax

from clrs._src import dfa_sampler
from clrs._src import yzd_utils
from clrs._src import dfa_nets
from clrs._src import dfa_processors


def FeedbackListGenerator(dfa_sampler: dfa_sampler.DFASampler,
                          batch_size: int):
    while True:
        dfa_sampler.next(batch_size=batch_size)


def forward_dfa_nets(processor_params_dict,
                     dfa_net_params_dict):
    test_processor_factory = dfa_processors.get_dfa_processor_factory(**processor_params_dict)
    return dfa_nets.DFANet(**dfa_net_params_dict,
                           processor_factory=test_processor_factory,
                           hint_repred_mode='soft',
                           nb_dims=None,  # this is used by categorical decoder
                           name='dfa_net')


if __name__ == '__main__':
    test_params_savepath = '/Users/yizhidou/Documents/ProGraMLTestPlayground/TestOutputFiles/poj104_103/test_params/test_params_v0.json'
    test_params_dict = yzd_utils.parse_params(params_filepath=test_params_savepath)
    test_sample_path_processor = yzd_utils.SamplePathProcessor(**test_params_dict['sample_path_processor'])
    test_sample_loader = yzd_utils.SampleLoader(sample_path_processor=test_sample_path_processor,
                                                **test_params_dict['sample_loader'])
    test_sampler = dfa_sampler.DFASampler(task_name=test_params_dict['task']['task_name'],
                                          sample_id_list=test_params_dict['dfa_sampler']['sample_id_list'],
                                          seed=test_params_dict['dfa_sampler']['seed'],
                                          sample_loader=test_sample_loader)

    feedback_generator = FeedbackListGenerator(dfa_sampler=test_sampler,
                                               batch_size=test_params_dict['dfa_sampler']['batch_size'])
    feedback_list = [next(feedback_generator) for _ in range(2)]
