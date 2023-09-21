import haiku as hk
import jax

from clrs._src import dfa_sampler
from clrs._src import yzd_utils
from clrs._src import dfa_nets
from clrs._src import dfa_processors


def FeedbackListGenerator(dfa_sampler: dfa_sampler.DFASampler,
                          batch_size: int = 1):
    while True:
        dfa_sampler.next(batch_size=batch_size)


def forward_dfa_nets():
    test_processor_factory = dfa_processors.get_dfa_processor_factory(kind='',
                                                                      nb_heads=0,
                                                                      activation=None,
                                                                      residual=True,
                                                                      use_ln=True)
    return dfa_nets.DFANet(spec=None,
                           hidden_dim=0,
                           encode_hints=True,
                           decode_hints=True,
                           processor_factory=test_processor_factory,
                           use_lstm=True,
                           encoder_init='',
                           dropout_prob=0,
                           hint_teacher_forcing=0,
                           hint_repred_mode='soft',
                           nb_dims=None,    # this is used by categorical decoder
                           nb_msg_passing_steps=5,
                           name='dfa_net')


if __name__ == '__main__':
    test_sample_path_processor = yzd_utils.SamplePathProcessor(sourcegraph_dir='',
                                                               errorlog_savepath='',
                                                               dataset_savedir='',
                                                               statistics_savepath='')
    test_sample_loader = yzd_utils.SampleLoader(sample_path_processor=test_sample_path_processor,
                                                max_iteration=0,
                                                max_num_pp=0,
                                                gkt_edges_rate=0,
                                                selected_num_ip=0)
    test_sampler = dfa_sampler.DFASampler(task_name='',
                                          sample_id_list=[],
                                          seed=0,
                                          sample_loader=test_sample_loader)

    feedback_generator = FeedbackListGenerator(dfa_sampler=test_sampler)
    feedback_list = [next(feedback_generator) for _ in range(2)]
