import jax
import random
import haiku as hk
import numpy as np
from clrs._src import dfa_samplers
from clrs._src import dfa_utils
from clrs._src import dfa_processors
from clrs._src import dfa_baselines

params_savepath = '/data_hdd/lx20/yzd_workspace/Params/params_poj104/poj_params_dfa_v1.json'
log_savepath = '/data_hdd/lx20/yzd_workspace/Logs/TrainLogs/poj_test.log'
params_dict = dfa_utils.parse_params(params_filepath=params_savepath)
sample_path_processor = dfa_utils.SamplePathProcessor(**params_dict['sample_path_processor'])
sample_loader = dfa_utils.SampleLoader(sample_path_processor=sample_path_processor,
                                       **params_dict['sample_loader'])

random.seed(params_dict['task']['seed'])
rng = np.random.RandomState(seed=params_dict['task']['seed'])
test_sampler = dfa_samplers.DFASampler(task_name=params_dict['task']['task_name'],
                                       sample_id_list=params_dict['dfa_sampler']['test_sample_id_list'],
                                       # seed=params_dict['task']['seed'],
                                       seed=rng.randint(2 ** 32),
                                       sample_loader=sample_loader,
                                       num_samples=params_dict['task']['num_samples_test_set'])
test_feedback_generator = dfa_samplers.FeedbackGenerator(dfa_sampler=test_sampler,
                                                         batch_size=params_dict['dfa_sampler']['batch_size'])
# feedback_list = [next(train_feedback_generator) for _ in range(1)]
# exit(666)
processor_factory = dfa_processors.get_dfa_processor_factory(**params_dict['processor'])

dfa_baseline_model = dfa_baselines.DFABaselineModel(processor_factory=processor_factory,
                                                    **params_dict['dfa_net'],
                                                    **params_dict['baseline_model'])
# print('doudou_test line 37 dfa_baseline_model __init__ done!')
# if os.path.isfile(os.path.join(dfa_baseline_model.checkpoint_path, params_dict['task']['model_save_name'])):
#     dfa_baseline_model.restore_model(file_name=params_dict['task']['model_save_name'])
#     print('the params are restored from ckpt!')
# else:
#     dfa_baseline_model.init(features=next(train_feedback_generator).features,
#                             seed=params_dict['task']['seed'])
#     print('no ckpt detected! so we init params from scratch!')
dfa_baseline_model.init(features=next(test_feedback_generator).features,
                        seed=params_dict['task']['seed'] + 1)
# print('dfa_train line 49 dfa_baseline_model init done!')
# exit(666)
epoch_idx = 0
log_str = ''
rng_key = jax.random.PRNGKey(rng.randint(2 ** 32))
while epoch_idx < params_dict['task']['nb_epochs']:
    # validate
    vali_sampler_this_epoch = dfa_samplers.DFASampler(task_name=params_dict['task']['task_name'],
                                                      sample_id_list=params_dict['dfa_sampler'][
                                                          'vali_sample_id_list'],
                                                      # seed=params_dict['task']['seed'],
                                                      seed=rng.randint(2 ** 32),
                                                      sample_loader=sample_loader,
                                                      num_samples=params_dict['task']['num_samples_vali_set'])
    vali_feedback_generator_this_epoch = dfa_samplers.FeedbackGenerator(dfa_sampler=vali_sampler_this_epoch,
                                                                        batch_size=params_dict['dfa_sampler'][
                                                                            'batch_size'],
                                                                        if_vali_or_test=True)
    vali_batch_idx, vali_precision_accum, vali_recall_accum, vali_f1_accum = 0.0, 0.0, 0.0, 0.0
    pos_num_accum, total_num_accum = 0.0, 0.0
    while True:
        try:
            vali_feedback_batch = next(vali_feedback_generator_this_epoch)
        except:
            break
        # batch_vali_loss = dfa_baseline_model.compute_loss(
        #         rng_key=jax.random.PRNGKey(params_dict['task']['seed']),
        #         feedback=vali_feedback_batch)
        # new_log_str = f'epoch_{epoch_idx} vali_batch {int(vali_batch_idx)}: loss = {batch_vali_loss}\n'
        new_rng_key, rng_key = jax.random.split(rng_key)
        vali_precision, vali_recall, vali_f1, positive_num, total_num = dfa_baseline_model.get_measures(
            rng_key=new_rng_key,
            feedback=vali_feedback_batch)
        new_log_str = f'vali epoch_{epoch_idx}_batch_{int(vali_batch_idx)}: precision = {vali_precision}; recall = {vali_recall}; F1 = {vali_f1}\npositive_num = {positive_num}; total_num = {total_num}; pos_percentage = {float(positive_num) / float(total_num)}\n'
        print(new_log_str, end='')
        log_str += new_log_str
        vali_precision_accum += vali_precision
        vali_recall_accum += vali_recall
        vali_f1_accum += vali_f1
        pos_num_accum += positive_num
        total_num_accum += total_num
        vali_batch_idx += 1

    new_log_str = f'vali epoch_{epoch_idx}: precision = {vali_precision_accum / vali_batch_idx}; recall = {vali_recall_accum / vali_batch_idx}; F1 = {vali_f1_accum / vali_batch_idx}\npositive_num = {pos_num_accum / vali_batch_idx}; total_num = {total_num_accum / vali_batch_idx}; pos_percentage = {pos_num_accum / total_num_accum}\n'
    print(new_log_str, end='')
    log_str += new_log_str
    with open(log_savepath, 'a') as log_recorder:
        log_recorder.write(log_str)
    del log_str
    log_str = ''
    # train
    train_batch_idx = 0.0
    train_loss_accum, train_precision_accum, train_recall_accum, train_f1_accum = 0.0, 0.0, 0.0, 0.0
    pos_num_accum, total_num_accum = 0.0, 0.0
    train_sampler = dfa_samplers.DFASampler(task_name=params_dict['task']['task_name'],
                                            sample_id_list=params_dict['dfa_sampler']['train_sample_id_list'],
                                            seed=rng.randint(2 ** 32),
                                            sample_loader=sample_loader,
                                            num_samples=params_dict['task']['num_samples_train_set'])
    train_feedback_generator = dfa_samplers.FeedbackGenerator(dfa_sampler=train_sampler,
                                                              batch_size=params_dict['dfa_sampler'][
                                                                  'batch_size'])
    while True:
        try:
            train_feedback_batch = next(train_feedback_generator)
            new_rng_key, rng_key = jax.random.split(rng_key)
            batch_train_loss = dfa_baseline_model.feedback(rng_key=new_rng_key,
                                                           feedback=train_feedback_batch)
            new_rng_key, rng_key = jax.random.split(rng_key)
            train_precision, train_recall, train_f1, positive_num, total_num = dfa_baseline_model.get_measures(
                rng_key=new_rng_key,
                feedback=train_feedback_batch)
            new_log_str = f'train epoch_{epoch_idx}_batch_{int(train_batch_idx)}: loss = {batch_train_loss}; precision = {train_precision}; recall = {train_recall}; F1 = {train_f1}\npositive_num = {positive_num}; total_num = {total_num}; pos_percentage = {float(positive_num) / float(total_num)}\n'
            print(new_log_str, end='')
            log_str += new_log_str
            train_loss_accum += batch_train_loss
            train_precision_accum += train_precision
            train_recall_accum += train_recall
            train_f1_accum += train_f1
            pos_num_accum += positive_num
            total_num_accum += total_num
            train_batch_idx += 1
        except:
            break
    new_log_str = f'train epoch_{epoch_idx}: loss = {train_loss_accum / train_batch_idx}; precision = {train_precision_accum / train_batch_idx}; recall = {train_recall_accum / train_batch_idx}; F1 = {train_f1_accum / train_batch_idx}\npositive_num = {pos_num_accum / train_batch_idx}; total_num = {total_num_accum / train_batch_idx}; pos_percentage = {pos_num_accum / total_num_accum}\n'
    print(new_log_str, end='')
    log_str += new_log_str
    with open(log_savepath, 'a') as log_recorder:
        log_recorder.write(log_str)
    del log_str
    log_str = ''
    epoch_idx += 1
    # save the model
    dfa_baseline_model.save_model(file_name=params_dict['task']['model_save_name'])
    print('the model has been saved~')
