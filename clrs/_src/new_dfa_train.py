import jax
import argparse
# import hashlib
import numpy as np
import os
from clrs._src import dfa_samplers
from clrs._src import dfa_utils
from clrs._src import dfa_processors
from clrs._src import dfa_baselines


def train(params_savedir, params_filename,
          statistics_filepath, if_clear, if_log):
    params_hash, params_savepath = dfa_utils.rename_params_file(params_savedir, params_filename,
                                                                if_test_params=False)
    print(f'We are dealing with {params_hash}...')
    params_dict = dfa_utils.parse_params(params_hash=params_hash,
                                         params_filepath=params_savepath,
                                         statistics_filepath=statistics_filepath,
                                         for_model_test=False)
    log_savepath = os.path.join(params_dict['log']['log_savedir'], f'{params_hash}.log')
    train_used_sample_ids_log_savepaht = os.path.join(params_dict['log']['used_sample_ids_savedir'], f'{params_hash}.used_samples')
    train_used_sample_ids = dict()
    if os.path.isfile(log_savepath) and if_clear and if_log:
        os.system(f'rm {log_savepath}')
        print('old log has been removed!')
    if os.path.isfile(log_savepath) and not if_clear and if_log:
        print('Previous log exist! Please specify if clear the previous log or not!')
        exit(58)

    sample_path_processor = dfa_utils.SamplePathProcessor(**params_dict['sample_path_processor'])
    rng = np.random.RandomState(seed=params_dict['task']['seed'])
    train_sample_loader = dfa_utils.SampleLoader(sample_path_processor=sample_path_processor,
                                                 seed=rng.randint(2 ** 32),
                                                 **params_dict['train_sample_loader'])
    vali_sample_loader = dfa_utils.SampleLoader(sample_path_processor=sample_path_processor,
                                                seed=rng.randint(2 ** 32),
                                                **params_dict['vali_sample_loader'])

    # sample_id_savedir = '/data_hdd/lx20/yzd_workspace/Logs/SampleIdLogs'
    train_sampler = dfa_samplers.DFASampler(task_name=params_dict['task']['task_name'],
                                            sample_id_list=params_dict['dfa_sampler']['train_sample_id_list'],
                                            seed=rng.randint(2 ** 32),
                                            sample_loader=train_sample_loader,
                                            # sample_id_savepath=os.path.join(sample_id_savedir,f'{params_hash}.train_sample_ids')
                                            )
    train_feedback_generator = dfa_samplers.FeedbackGenerator_all_tasks(dfa_sampler=train_sampler,
                                                                        batch_size=params_dict['dfa_sampler'][
                                                                            'batch_size'])
    vali_sampler = dfa_samplers.DFASampler(task_name=params_dict['task']['task_name'],
                                           sample_id_list=params_dict['dfa_sampler'][
                                               'vali_sample_id_list'],
                                           # seed=params_dict['task']['seed'],
                                           seed=rng.randint(2 ** 32),
                                           sample_loader=vali_sample_loader,
                                           # sample_id_savepath=os.path.join(sample_id_savedir, f'{params_hash}.vali_sample_ids')
                                           )
    vali_feedback_generator = dfa_samplers.FeedbackGenerator_all_tasks(dfa_sampler=vali_sampler,
                                                                       batch_size=params_dict['dfa_sampler'][
                                                                           'batch_size'])
    # exit(666)
    processor_factory = dfa_processors.get_dfa_processor_factory(**params_dict['processor'])
    dfa_baseline_model = dfa_baselines.DFABaselineModel(processor_factory=processor_factory,
                                                        # version_of_DFANet=version_of_DFANet,
                                                        **params_dict['dfa_net'],
                                                        **params_dict['baseline_model'])
    _, init_feedback = next(train_feedback_generator)
    dfa_baseline_model.init(features=init_feedback.features,
                            seed=params_dict['task']['seed'] + 1)
    # print('dfa_train line 116 dfa_baseline_model init done!')
    # exit(666)
    epoch_idx = 0
    train_step_per_epoch = params_dict['task']['num_samples_train_set']
    vali_step_per_epoch = params_dict['task']['num_samples_vali_set']
    iterate_entire_train_set = True if train_step_per_epoch < 0 else False
    iterate_entire_vali_set = True if vali_step_per_epoch < 0 else False
    if iterate_entire_train_set:
        train_sampler.reset_sample_id_iter()
    if iterate_entire_vali_set:
        vali_sampler.reset_sample_id_iter()
    log_str = ''
    train_sampled_id_remain = ''
    vali_sampled_id_remain = ''
    rng_key = jax.random.PRNGKey(rng.randint(2 ** 32))
    while epoch_idx < params_dict['task']['nb_epochs']:
        # validate
        vali_batch_idx, vali_precision_accum, vali_recall_accum, vali_f1_accum = 0.0, 0.0, 0.0, 0.0
        num_vali_batch_liveness, liveness_vali_precision_accum, liveness_vali_recall_accum, liveness_vali_f1_accum = 0.0, 0.0, 0.0, 0.0
        num_vali_batch_dominance, dominance_vali_precision_accum, dominance_vali_recall_accum, dominance_vali_f1_accum = 0.0, 0.0, 0.0, 0.0
        num_vali_batch_reachability, reachability_vali_precision_accum, reachability_vali_recall_accum, reachability_vali_f1_accum = 0.0, 0.0, 0.0, 0.0
        # pos_num_accum, total_num_accum = 0.0, 0.0
        while True:
            if iterate_entire_vali_set:
                try:
                    sampled_ids_this_batch, task_name_for_this_batch, vali_feedback_batch = next(vali_feedback_generator)
                except RuntimeError:  # Stop
                    break
            else:
                if vali_batch_idx == vali_step_per_epoch:
                    break
                sampled_ids_this_batch, task_name_for_this_batch, vali_feedback_batch = next(vali_feedback_generator)
            cur_vali_sampled_id = sampled_ids_this_batch[0]
            if not cur_vali_sampled_id == vali_sampled_id_remain:
                new_log_str = f'\nvali epoch_{epoch_idx}_batch_{int(vali_batch_idx)} {cur_vali_sampled_id}: '
                print(new_log_str)
                log_str += new_log_str
                vali_sampled_id_remain = cur_vali_sampled_id
                vali_batch_idx += 1
        # while vali_batch_idx < vali_step_per_epoch:
        #     task_name_for_this_batch, vali_feedback_batch = next(vali_feedback_generator)
            new_rng_key, rng_key = jax.random.split(rng_key)
            vali_precision, vali_recall, vali_f1, _, _ = dfa_baseline_model.get_measures(
                rng_key=new_rng_key,
                feedback=vali_feedback_batch)
            # new_log_str = f'vali epoch_{epoch_idx}_batch_{int(vali_batch_idx)} ({task_name_for_this_batch}): precision = {vali_precision}; recall = {vali_recall}; F1 = {vali_f1}\n'
            new_log_str = f'{task_name_for_this_batch}: p = {vali_precision:.4f}; r = {vali_recall:.4f}; f1 = {vali_f1:.4f}. '
            print(new_log_str)
            log_str += new_log_str
            if task_name_for_this_batch == 'liveness':
                liveness_vali_precision_accum += vali_precision
                liveness_vali_recall_accum += vali_recall
                liveness_vali_f1_accum += vali_f1
                num_vali_batch_liveness += 1
            elif task_name_for_this_batch == 'dominance':
                dominance_vali_precision_accum += vali_precision
                dominance_vali_recall_accum += vali_recall
                dominance_vali_f1_accum += vali_f1
                num_vali_batch_dominance += 1
            elif task_name_for_this_batch == 'reachability':
                reachability_vali_precision_accum += vali_precision
                reachability_vali_recall_accum += vali_recall
                reachability_vali_f1_accum += vali_f1
                num_vali_batch_reachability += 1
            else:
                print('dfa_train line 150, unrecognized task_name!!')
                assert False
        liveness_log_str = f'vali epoch_{epoch_idx}: {int(num_vali_batch_liveness)} liveness batches; p = {(liveness_vali_precision_accum / num_vali_batch_liveness):.4f}; r = {(liveness_vali_recall_accum / num_vali_batch_liveness):.4f}; F1 = {(liveness_vali_f1_accum / num_vali_batch_liveness):.4f}\n'
        dominance_log_str = f'vali epoch_{epoch_idx}: {int(num_vali_batch_dominance)} dominance batches; p = {(dominance_vali_precision_accum / num_vali_batch_dominance):.4f}; r = {(dominance_vali_recall_accum / num_vali_batch_dominance):.4f}; F1 = {(dominance_vali_f1_accum / num_vali_batch_dominance):.4f}\n'
        reachability_log_str = f'vali epoch_{epoch_idx}: {int(num_vali_batch_reachability)} reachability batches; p = {(reachability_vali_precision_accum / num_vali_batch_reachability):.4f}; r = {(reachability_vali_recall_accum / num_vali_batch_reachability):.4f}; F1 = {(reachability_vali_f1_accum / num_vali_batch_reachability):.4f}\n'
        new_log_str = liveness_log_str + reachability_log_str + dominance_log_str
        print(new_log_str, end='')
        log_str += new_log_str
        if if_log:
            with open(log_savepath, 'a') as log_recorder:
                log_recorder.write(log_str)
            if epoch_idx == 0:
                dfa_baseline_model.save_model(file_name=f'{params_hash}.epoch_{0}')
                print('the model (untrained) has been saved~')
        del log_str
        log_str = ''
        if iterate_entire_vali_set:
            vali_sampler.reset_sample_id_iter()
        # train
        train_batch_idx = 0.0
        train_loss_accum, train_precision_accum, train_recall_accum, train_f1_accum = 0.0, 0.0, 0.0, 0.0
        # pos_num_accum, total_num_accum = 0.0, 0.0
        num_train_batch_liveness, liveness_train_loss_accum, liveness_train_precision_accum, liveness_train_recall_accum, liveness_train_f1_accum = 0.0, 0.0, 0.0, 0.0, 0.0
        num_train_batch_dominance, dominance_train_loss_accum, dominance_train_precision_accum, dominance_train_recall_accum, dominance_train_f1_accum = 0.0, 0.0, 0.0, 0.0, 0.0
        num_train_batch_reachability, reachability_train_loss_accum, reachability_train_precision_accum, reachability_train_recall_accum, reachability_train_f1_accum = 0.0, 0.0, 0.0, 0.0, 0.0
        while True:
            if iterate_entire_train_set:
                try:
                    sampled_ids_this_batch, task_name_for_this_batch, train_feedback_batch = next(train_feedback_generator)
                except RuntimeError:  # Stop
                    break
            else:
                if train_batch_idx == train_step_per_epoch:
                    break
                sampled_ids_this_batch, task_name_for_this_batch, train_feedback_batch = next(train_feedback_generator)
            cur_train_sampled_id = sampled_ids_this_batch[0]
            if not cur_train_sampled_id == train_sampled_id_remain:
                new_log_str = f'train epoch_{epoch_idx}_batch_{int(train_batch_idx)} {cur_train_sampled_id}: '
                print(new_log_str)
                log_str += new_log_str
                train_sampled_id_remain = cur_train_sampled_id
                train_used_sample_ids[cur_train_sampled_id] = 1
                train_batch_idx += 1
        # while train_batch_idx < train_step_per_epoch:
        #     task_name_for_this_batch, train_feedback_batch = next(train_feedback_generator)
            new_rng_key, rng_key = jax.random.split(rng_key)
            batch_train_loss = dfa_baseline_model.feedback(rng_key=new_rng_key,
                                                           feedback=train_feedback_batch)
            new_rng_key, rng_key = jax.random.split(rng_key)
            train_precision, train_recall, train_f1 = dfa_baseline_model.get_measures(
                rng_key=new_rng_key,
                feedback=train_feedback_batch)
            # new_log_str = f'train epoch_{epoch_idx}_batch_{int(train_batch_idx)} ({task_name_for_this_batch}): loss = {batch_train_loss}; precision = {train_precision}; recall = {train_recall}; F1 = {train_f1}\n'
            new_log_str = f'{task_name_for_this_batch}: loss = {batch_train_loss:.6f}; p = {train_precision:.4f}; recall = {train_recall:.4f}; F1 = {train_f1:.4f}. '
            print(new_log_str)
            log_str += new_log_str
            if task_name_for_this_batch == 'liveness':
                liveness_train_loss_accum += batch_train_loss
                liveness_train_precision_accum += train_precision
                liveness_train_recall_accum += train_recall
                liveness_train_f1_accum += train_f1
                num_train_batch_liveness += 1
            elif task_name_for_this_batch == 'dominance':
                dominance_train_loss_accum += batch_train_loss
                dominance_train_precision_accum += train_precision
                dominance_train_recall_accum += train_recall
                dominance_train_f1_accum += train_f1
                num_train_batch_dominance += 1
            elif task_name_for_this_batch == 'reachability':
                reachability_train_loss_accum += batch_train_loss
                reachability_train_precision_accum += train_precision
                reachability_train_recall_accum += train_recall
                reachability_train_f1_accum += train_f1
                num_train_batch_reachability += 1
            else:
                print('dfa_train line 230, unrecognized task_name!!')
                assert False
        liveness_log_str = f'train epoch_{epoch_idx}: {num_train_batch_liveness} liveness batches; loss = {(liveness_train_loss_accum / num_train_batch_liveness):.4f}; p = {(liveness_train_precision_accum / num_train_batch_liveness):.4f}; r = {(liveness_train_recall_accum / num_train_batch_liveness):.4f}; F1 = {(liveness_train_f1_accum / num_train_batch_liveness):.4f}\n'
        dominance_log_str = f'train epoch_{epoch_idx}: {num_train_batch_dominance} dominance batches; loss = {(dominance_train_loss_accum / num_train_batch_dominance):.4f}; pr = {(dominance_train_precision_accum / num_train_batch_dominance):.4f}; r = {(dominance_train_recall_accum / num_train_batch_dominance):.4f}; F1 = {(dominance_train_f1_accum / num_train_batch_dominance):.4f}\n'
        reachability_log_str = f'train epoch_{epoch_idx}: {num_train_batch_reachability} reachability batches; loss = {(reachability_train_loss_accum / num_train_batch_reachability):.4f} p = {(reachability_train_precision_accum / num_train_batch_reachability):.4f}; r = {(reachability_train_recall_accum / num_train_batch_reachability):.4f}; F1 = {(reachability_train_f1_accum / num_train_batch_reachability):.4f}\n'
        new_log_str = liveness_log_str + reachability_log_str + dominance_log_str

        print(new_log_str, end='')
        log_str += new_log_str
        if if_log:
            with open(log_savepath, 'a') as log_recorder:
                log_recorder.write(log_str)
        del log_str
        log_str = ''
        epoch_idx += 1
        # save the model
        if if_log:
            dfa_baseline_model.save_model(file_name=f'{params_hash}.epoch_{epoch_idx}')
            print('the model has been saved~')
        # log errored sample ids
        sample_path_processor.dump_errored_samples_to_log()


# def test():


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Please input the params filename')
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument('--clear', action='store_true')
    parser.add_argument('--unlog', action='store_true')
    args = parser.parse_args()
    params_savedir = '/data_hdd/lx20/yzd_workspace/Params/TrainParamsPOJ104/'
    statistics_filepath = '/data_hdd/lx20/yzd_workspace/Datasets/Statistics/POJ104Statistics/poj104_statistics.json'
    if not os.path.isfile(os.path.join(params_savedir, args.params)):
        print('the specified params does not exist!')
        exit(176)
    # params_filename = 'params_not_hint_as_outpt.json'
    train(params_savedir=params_savedir,
          params_filename=args.params,
          statistics_filepath=statistics_filepath,
          if_clear=args.clear,
          if_log=not args.unlog)
