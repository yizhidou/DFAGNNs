import haiku as hk
import jax
import os
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
                                       seed=params_dict['task']['seed'],
                                       sample_loader=sample_loader)
vali_sampler = dfa_sampler.DFASampler(task_name=params_dict['task']['task_name'],
                                      sample_id_list=params_dict['dfa_sampler']['vali_sample_id_list'],
                                      seed=params_dict['task']['seed'],
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

if os.path.isfile(os.path.join(dfa_baseline_model.checkpoint_path, params_dict['task']['model_save_name'])):
    dfa_baseline_model.restore_model(file_name=params_dict['task']['model_save_name'])
    print('the params are restored from ckpt!')
else:
    dfa_baseline_model.init(features=next(train_feedback_generator).features,
                            seed=params_dict['task']['seed'])
    print('no ckpt detected! so we init params from scratch!')
epoch_idx = 0
while epoch_idx < params_dict['task']['nb_epochs']:
    # validate
    vali_feedback_generator = dfa_sampler.FeedbackGenerator_limited(dfa_sampler=vali_sampler,
                                                                    batch_size=params_dict['dfa_sampler']['batch_size'])
    vali_batch_idx = 0.0
    vali_loss = 0
    for vali_feedback_batch in vali_feedback_generator:
        batch_vali_loss = dfa_baseline_model.compute_loss(
            rng_key=jax.random.PRNGKey(params_dict['task']['seed']),
            feedback=vali_feedback_batch)
        print(f'vali_batch {vali_batch_idx}: loss = {batch_vali_loss}')
        vali_loss += batch_vali_loss
        vali_batch_idx += 1
    print(f'epoch {epoch_idx}: mean vali_loss = {vali_loss / vali_batch_idx}')
    # train
    train_batch_idx = 0.0
    train_loss = 0
    while train_batch_idx < params_dict['task']['nb_training_steps']:
        train_feedback_batch = next(train_feedback_generator)
        batch_train_loss = dfa_baseline_model.feedback(rng_key=jax.random.PRNGKey(params_dict['task']['seed']),
                                                       feedback=train_feedback_batch)
        print(f'train_batch {train_batch_idx}: loss = {batch_train_loss}')
        train_loss += batch_train_loss
        train_batch_idx += 1
    print(f'epoch {epoch_idx}: mean train_loss = {train_loss / train_batch_idx}')
    epoch_idx += 1
    # save the model
    dfa_baseline_model.save_model(file_name=params_dict['task']['model_save_name'])
    print('the model has been saved~')
