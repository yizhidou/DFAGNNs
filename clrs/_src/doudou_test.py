import haiku as hk
import jax

from clrs._src import dfa_sampler
from clrs._src import dfa_utils
from clrs._src import dfa_processors
from clrs._src import dfa_baselines

params_savepath = '/Users/yizhidou/Documents/ProGraMLTestPlayground/TestOutputFiles/poj104_103/test_params/test_params_v0.json'
params_dict = dfa_utils.parse_params(params_filepath=params_savepath)
sample_path_processor = dfa_utils.SamplePathProcessor(**params_dict['sample_path_processor'])
sample_loader = dfa_utils.SampleLoader(sample_path_processor=sample_path_processor,
                                       **params_dict['sample_loader'])

train_sampler = dfa_sampler.DFASampler(task_name=params_dict['task']['task_name'],
                                       sample_id_list=params_dict['dfa_sampler']['train_sample_id_list'],
                                       seed=params_dict['dfa_sampler']['seed'],
                                       sample_loader=sample_loader)
vali_sampler = dfa_sampler.DFASampler(task_name=params_dict['task']['task_name'],
                                      sample_id_list=params_dict['dfa_sampler']['vali_sample_id_list'],
                                      seed=params_dict['dfa_sampler']['seed'],
                                      sample_loader=sample_loader)
test_sampler = dfa_sampler.DFASampler(task_name=params_dict['task']['task_name'],
                                      sample_id_list=params_dict['dfa_sampler']['test_sample_id_list'],
                                      seed=params_dict['task']['seed'],
                                      sample_loader=sample_loader)

train_feedback_generator = dfa_sampler.FeedbackGenerator(dfa_sampler=train_sampler,
                                                         batch_size=params_dict['dfa_sampler']['batch_size'])
# feedback_list = [next(feedback_generator) for _ in range(5)]

processor_factory = dfa_processors.get_dfa_processor_factory(**params_dict['processor'])

dfa_baseline_model = dfa_baselines.DFABaselineModel(processor_factory=processor_factory,
                                                    **params_dict['dfa_net'],
                                                    **params_dict['baseline_model'])
dfa_baseline_model.init(features=next(train_feedback_generator).features,
                        seed=params_dict['task']['seed'])

# validate
vali_feedback_generator = dfa_sampler.FeedbackGenerator_limited(dfa_sampler=vali_sampler,
                                                                batch_size=params_dict['dfa_sampler']['batch_size'])
vali_batch_idx = 0
vali_loss = 0
for vali_feedback in vali_feedback_generator:
    vali_loss += dfa_baseline_model.compute_loss(rng_key=jax.random.PRNGKey(42),
                                                 feedback=vali_feedback)
