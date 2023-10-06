import haiku as hk
import jax

from clrs._src import dfa_sampler
from clrs._src import dfa_utils
from clrs._src import dfa_nets
from clrs._src import dfa_processors
from clrs._src import dfa_baselines


def FeedbackListGenerator(dfa_sampler: dfa_sampler.DFASampler,
                          batch_size: int):
    while True:
        yield dfa_sampler.next(batch_size=batch_size)


def _dfa_nets(processor_params_dict,
              dfa_net_params_dict,
              features_list,
              repred: bool,
              algorithm_index: int,
              return_hints: bool,
              return_all_outputs: bool
              ):
    test_processor_factory = dfa_processors.get_dfa_processor_factory(**processor_params_dict)
    _net = dfa_nets.DFANet(**dfa_net_params_dict,
                           processor_factory=test_processor_factory)
    return _net(features_list=features_list,
                repred=repred,
                algorithm_index=algorithm_index,
                return_hints=return_hints,
                return_all_outputs=return_all_outputs)


if __name__ == '__main__':
    test_params_savepath = '/Users/yizhidou/Documents/ProGraMLTestPlayground/TestOutputFiles/poj104_103/test_params/test_params_v0.json'
    test_params_dict = dfa_utils.parse_params(params_filepath=test_params_savepath)
    test_sample_path_processor = dfa_utils.SamplePathProcessor(**test_params_dict['sample_path_processor'])
    test_sample_loader = dfa_utils.SampleLoader(sample_path_processor=test_sample_path_processor,
                                                **test_params_dict['sample_loader'])

    # test_sample_loader.load_a_sample(task_name='dfa_dominance',
    #                                  sample_id='poj104_103.12081.4')

    test_sampler = dfa_sampler.DFASampler(task_name=test_params_dict['task']['task_name'],
                                          sample_id_list=test_params_dict['dfa_sampler']['sample_id_list'],
                                          seed=test_params_dict['dfa_sampler']['seed'],
                                          sample_loader=test_sample_loader)

    feedback_generator = FeedbackListGenerator(dfa_sampler=test_sampler,
                                               batch_size=test_params_dict['dfa_sampler']['batch_size'])
    feedback_list = [next(feedback_generator) for _ in range(5)]

    test_processor_factory = dfa_processors.get_dfa_processor_factory(**test_params_dict['processor'])

    test_dfa_baseline = dfa_baselines.DFABaselineModel(dummy_trajectory=[feedback_list[0]],
                                                       processor_factory=test_processor_factory,
                                                       **test_params_dict['dfa_net'],
                                                       **test_params_dict['baseline_model'])
    test_dfa_baseline.init(features=[feedback_list[1].features],
                           seed=6)
    loss1 = test_dfa_baseline.feedback(rng_key=jax.random.PRNGKey(42),
                                       feedback=feedback_list[2])
    loss2 = test_dfa_baseline.feedback(rng_key=jax.random.PRNGKey(42),
                                       feedback=feedback_list[3])

    print(f'the final loss1 = {loss1}; loss2 = {loss2}')

    # def _use_net(features_list,
    #              repred,
    #              algorithm_index,
    #              return_hints,
    #              return_all_outputs):
    #     return dfa_nets.DFANet(processor_factory=test_processor_factory,
    #                            **test_params_dict['dfa_net'])(features_list=features_list,
    #                                                           repred=repred,
    #                                                           algorithm_index=algorithm_index,
    #                                                           return_hints=return_hints,
    #                                                           return_all_outputs=return_all_outputs)
    #
    #
    # net_fn = hk.transform(_use_net)
    #
    # print(f'the type of transformed_net is: {type(net_fn)}')
    # feature_list = [feedback_list[0].features]
    # params = net_fn.init(rng=jax.random.PRNGKey(42),
    #                      features_list=feature_list,
    #                      repred=False,
    #                      algorithm_index=-1,
    #                      return_hints=True,
    #                      return_all_outputs=True)
