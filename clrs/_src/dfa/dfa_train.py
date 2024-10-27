import jax
import argparse
import json
import numpy as np
import os
from typing import List
from clrs._src import dfa_samplers
from clrs._src import dfa_utils
from clrs._src import dfa_processors
from clrs._src import dfa_baselines, dfa_specs


def rename_train_params_file(params_savedir,
                             params_filename):
    cur_params_filepath = os.path.join(params_savedir, params_filename)
    params_hash = dfa_utils.compute_hash(file_path=cur_params_filepath)

    new_params_filename = f'{params_hash}.train'
    params_filename_prefix = params_filename.split('.')[0]
    if params_filename_prefix == params_hash:
        assert params_filename == new_params_filename
        return params_hash, cur_params_filepath
    new_params_filepath = os.path.join(params_savedir, new_params_filename)
    os.system(f'mv {cur_params_filepath} {new_params_filepath}')
    if not os.path.isfile(new_params_filepath):
        print('where is the renamed params file???')
        exit(1)
    return params_hash, new_params_filepath


def parse_train_params(params_hash: str,
                       params_filepath: str,
                       ):
    with open(params_filepath) as params_loader:
        params_dict = json.load(params_loader)
    if 'notice' in params_dict and params_dict['notice']['use_xla_flags']:
        os.environ['XLA_FLAGS'] = (
            '--xla_gpu_deterministic_ops=true '
            '--xla_gpu_autotune_level=0 '
        )
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
    with open(params_dict['sample_path_processor']['errorlog_savepath']) as errored_sample_ids_loader:
        errored_sample_ids = json.load(errored_sample_ids_loader)
    full_statistics_filepath = params_dict['log']['full_statistics_filepath']
    candi_sample_id_list = []
    if isinstance(params_dict['dfa_sampler']['train_sample_id_savepath'], List):
        for vali_sample_id_savepath in params_dict['dfa_sampler']['train_sample_id_savepath']:
            with open(vali_sample_id_savepath) as sample_ids_loader:
                for line in sample_ids_loader.readlines():
                    sample_id = line.strip()
                    candi_sample_id_list.append(sample_id)
    else:
        assert isinstance(params_dict['dfa_sampler']['train_sample_id_savepath'], str)
        with open(params_dict['dfa_sampler']['train_sample_id_savepath']) as sample_ids_loader:
            for line in sample_ids_loader.readlines():
                sample_id = line.strip()
                candi_sample_id_list.append(sample_id)
    params_dict['dfa_sampler']['train_sample_id_list'] = dfa_utils.filter_sample_list(
        full_statistics_savepath=full_statistics_filepath,
        errored_sample_ids=errored_sample_ids,
        max_num_pp=params_dict['train_sample_loader']['max_num_pp'],
        min_num_pp=params_dict['train_sample_loader']['min_num_pp'],
        cfg_edges_rate=params_dict['train_sample_loader']['cfg_edges_rate'],
        sample_ids=candi_sample_id_list)
    del params_dict['dfa_sampler']['train_sample_id_savepath']
    if params_dict['train_sample_loader']['dfa_version'] == 0:
        # params_dict['vali_sample_loader']['dfa_version'] = 0
        params_dict['dfa_net']['spec'] = [dfa_specs.DFASPECS['dfa']]
        params_dict['dfa_net']['dfa_version'] = 0
    elif params_dict['train_sample_loader']['dfa_version'] == 1:
        # params_dict['vali_sample_loader']['dfa_version'] = 1
        params_dict['dfa_net']['spec'] = [dfa_specs.DFASPECS['dfa_v1']]
        params_dict['dfa_net']['dfa_version'] = 1
    elif params_dict['train_sample_loader']['dfa_version'] == 2:
        # params_dict['vali_sample_loader']['dfa_version'] = 2
        params_dict['dfa_net']['spec'] = [dfa_specs.DFASPECS['dfa_v2']]
        params_dict['dfa_net']['dfa_version'] = 2
    elif params_dict['train_sample_loader']['dfa_version'] == 3:
        # params_dict['vali_sample_loader']['dfa_version'] = 2
        params_dict['dfa_net']['spec'] = [dfa_specs.DFASPECS['dfa_v3']]
        params_dict['dfa_net']['dfa_version'] = 3
    elif params_dict['train_sample_loader']['dfa_version'] == 4:
        # params_dict['vali_sample_loader']['dfa_version'] = 2
        params_dict['dfa_net']['spec'] = [dfa_specs.DFASPECS['dfa_v4']]
        params_dict['dfa_net']['dfa_version'] = 4
    else:
        assert params_dict['train_sample_loader']['dfa_version'] is None
        params_dict['dfa_net']['spec'] = [dfa_specs.DFASPECS[params_dict['task']['task_name']]]
        params_dict['dfa_net']['dfa_version'] = None
        params_dict['processor']['activation'] = dfa_utils._get_activation(params_dict['processor']['activation_name'])
        del params_dict['processor']['activation_name']
    params_dict['baseline_model']['checkpoint_path'] = os.path.join(params_dict['baseline_model']['checkpoint_path'],
                                                                    f'{params_hash}_ckpt')
    if not os.path.isdir(params_dict['baseline_model']['checkpoint_path']):
        os.system('mkdir {}'.format(params_dict['baseline_model']['checkpoint_path']))
    if params_dict['processor']['kind'] == 'DFAGNN_plus':
        params_dict['baseline_model']['plus_or_others'] = 'plus'
    elif params_dict['processor']['kind'] in ['DFAGNN', 'DFAGNN_minus']:
        params_dict['baseline_model']['plus_or_others'] = 'others'
    else:
        print('unrecognized version of GNN_kind!')
        raise dfa_utils.DFAException(dfa_utils.DFAException.UNRECOGNIZED_GNN_TYPE)
    assert params_dict['dfa_net']['dfa_version'] in ['plus', 'others']
    if 'just_one_layer' in params_dict['dfa_net'] and params_dict['dfa_net']['just_one_layer']:
        assert params_dict['train_sample_loader']['expected_trace_len'] == 2
    if not 'exclude_trace_loss' in params_dict['dfa_net']:
        params_dict['dfa_net']['exclude_trace_loss'] = False
    if params_dict['dfa_net']['exclude_trace_loss']:
        assert params_dict['dfa_net']['hint_teacher_forcing'] is None, "This training param decides not to use trace supervision, so the teacher forcing got to be canceled!"
    return params_dict


def train(params_savedir, params_filename,
          if_clear, if_log, model_save_gap):
    params_hash, params_savepath = rename_train_params_file(params_savedir, params_filename)
    print(f'We are dealing with {params_hash}...')
    params_dict = parse_train_params(params_hash=params_hash,
                                     params_filepath=params_savepath,
                                     )
    log_savepath = os.path.join(params_dict['log']['log_savedir'], f'{params_hash}.log')
    if os.path.isfile(log_savepath) and if_clear and if_log:
        os.system(f'rm {log_savepath}')
        print('old log has been removed!')
    if os.path.isfile(log_savepath) and not if_clear and if_log:
        assert False, 'Previous log exist! Please specify if clear the previous log or not!'


    sample_path_processor = dfa_utils.SamplePathProcessor(**params_dict['sample_path_processor'])
    rng = np.random.RandomState(seed=params_dict['task']['seed'])
    train_sample_loader = dfa_utils.SampleLoader(sample_path_processor=sample_path_processor,
                                                 seed=rng.randint(2 ** 32),
                                                 **params_dict['train_sample_loader'])
    
    train_step_per_epoch = params_dict['task']['num_samples_train_set']
    iterate_entire_train_set = True if train_step_per_epoch < 0 else False
    train_sampler = dfa_samplers.DFASampler(task_name=params_dict['task']['task_name'],
                                            sample_id_list=params_dict['dfa_sampler']['train_sample_id_list'],
                                            seed=rng.randint(2 ** 32),
                                            sample_loader=train_sample_loader,
                                            iterate_all=iterate_entire_train_set,
                                            # sample_id_savepath=os.path.join(sample_id_savedir,f'{params_hash}.train_sample_ids')
                                            )
    train_feedback_generator = dfa_samplers.FeedbackGenerator_all_tasks(dfa_sampler=train_sampler,
                                                                        batch_size=params_dict['dfa_sampler'][
                                                                            'batch_size'])
    processor_factory = dfa_processors.get_dfa_processor_factory(**params_dict['processor'])
    dfa_baseline_model = dfa_baselines.DFABaselineModel(processor_factory=processor_factory,
                                                        **params_dict['dfa_net'],
                                                        **params_dict['baseline_model'])
    _, _, init_feedback = next(train_feedback_generator)
    dfa_baseline_model.init(features=init_feedback.features,
                            seed=params_dict['task']['seed'] + 1)
    next(train_feedback_generator)
    next(train_feedback_generator)
    epoch_idx = 0
    log_str = ''
    train_sampled_id_remain = ''
    rng_key = jax.random.PRNGKey(rng.randint(2 ** 32))
    while epoch_idx < params_dict['task']['nb_epochs']:
        if epoch_idx == 0 and if_log:
            dfa_baseline_model.save_model(file_name=f'{params_hash}.epoch_{0}')
            print('the model (untrained) has been saved~')
        # train
        train_batch_idx = 0.0
        num_train_batch_liveness, liveness_train_loss_accum, liveness_train_precision_accum, liveness_train_recall_accum, liveness_train_f1_accum, liveness_trace_f1_accum = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        num_train_batch_dominance, dominance_train_loss_accum, dominance_train_precision_accum, dominance_train_recall_accum, dominance_train_f1_accum, dominance_trace_f1_accum = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        num_train_batch_reachability, reachability_train_loss_accum, reachability_train_precision_accum, reachability_train_recall_accum, reachability_train_f1_accum, reachability_trace_f1_accum = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        liveness_done, reachability_done, dominance_done = False, False, False
        while True:
            if iterate_entire_train_set:
                try:
                    sampled_ids_this_batch, task_name_for_this_batch, train_feedback_batch = next(
                        train_feedback_generator)
                except RuntimeError:  # Stop
                    break
            else:
                if (train_batch_idx == train_step_per_epoch) and liveness_done and reachability_done and dominance_done:
                    break
                sampled_ids_this_batch, task_name_for_this_batch, train_feedback_batch = next(train_feedback_generator)
            cur_train_sampled_id = sampled_ids_this_batch[0]
            if not cur_train_sampled_id == train_sampled_id_remain:
                new_log_str = f'\ntrain epoch_{epoch_idx}_batch_{int(train_batch_idx)} {cur_train_sampled_id}: '
                print(new_log_str)
                log_str += new_log_str
                train_sampled_id_remain = cur_train_sampled_id
                train_batch_idx += 1
                liveness_done, reachability_done, dominance_done = False, False, False
            new_rng_key, rng_key = jax.random.split(rng_key)

            batch_train_loss = dfa_baseline_model.feedback(rng_key=new_rng_key,
                                                            feedback=train_feedback_batch)
            new_rng_key, rng_key = jax.random.split(rng_key)
            trace_h_precision_list, trace_h_recall_list, trace_h_f1_list, train_precision, train_recall, train_f1 = dfa_baseline_model.get_measures(
                rng_key=new_rng_key,
                feedback=train_feedback_batch,
                return_hints=True,
                print_full_trace_h_f1_list=not if_log)
            if len(trace_h_f1_list) == 0:
                mean_trace_f1 = train_f1
            else:
                mean_trace_f1 = sum(trace_h_f1_list) / float(len(trace_h_f1_list))
            new_log_str = f'{task_name_for_this_batch}: mean_t_f1 = {mean_trace_f1:.4f}; loss = {batch_train_loss}; p = {train_precision:.4f}; recall = {train_recall:.4f}; F1 = {train_f1:.4f}. '
            print(new_log_str)
            log_str += new_log_str
            if task_name_for_this_batch == 'liveness':
                liveness_train_loss_accum += batch_train_loss
                liveness_train_precision_accum += train_precision
                liveness_train_recall_accum += train_recall
                liveness_train_f1_accum += train_f1
                liveness_trace_f1_accum += mean_trace_f1
                num_train_batch_liveness += 1
                liveness_done = True
            elif task_name_for_this_batch == 'reachability':
                reachability_train_loss_accum += batch_train_loss
                reachability_train_precision_accum += train_precision
                reachability_train_recall_accum += train_recall
                reachability_train_f1_accum += train_f1
                reachability_trace_f1_accum += mean_trace_f1
                num_train_batch_reachability += 1
                reachability_done = True
            elif task_name_for_this_batch == 'dominance':
                dominance_train_loss_accum += batch_train_loss
                dominance_train_precision_accum += train_precision
                dominance_train_recall_accum += train_recall
                dominance_train_f1_accum += train_f1
                dominance_trace_f1_accum += mean_trace_f1
                num_train_batch_dominance += 1
                dominance_done = True
            else:
                print('dfa_train line 230, unrecognized task_name!!')
                assert False
            if if_log and train_batch_idx % 500 == 0:
                with open(log_savepath, 'a') as log_recorder:
                    log_recorder.write(log_str)
                del log_str
                log_str = ''
        if num_train_batch_liveness == 0:
            print(f'epoch {epoch_idx} empty!!!, so skip')
            train_sampler.reset_sample_id_iter()
            train_feedback_generator = dfa_samplers.FeedbackGenerator_all_tasks(dfa_sampler=train_sampler,
                                                                                batch_size=params_dict['dfa_sampler'][
                                                                                    'batch_size'])
            continue
        liveness_log_str = f'\ntrain epoch_{epoch_idx}: {num_train_batch_liveness} liveness batches; loss = {(liveness_train_loss_accum / num_train_batch_liveness)}; mean_t_f1 = {(liveness_trace_f1_accum / num_train_batch_liveness):.4f}; p = {(liveness_train_precision_accum / num_train_batch_liveness):.4f}; r = {(liveness_train_recall_accum / num_train_batch_liveness):.4f}; F1 = {(liveness_train_f1_accum / num_train_batch_liveness):.4f}\n'
        dominance_log_str = f'train epoch_{epoch_idx}: {num_train_batch_dominance} dominance batches; loss = {(dominance_train_loss_accum / num_train_batch_dominance)}; mean_t_f1 = {(dominance_trace_f1_accum / num_train_batch_dominance):.4f}; pr = {(dominance_train_precision_accum / num_train_batch_dominance):.4f}; r = {(dominance_train_recall_accum / num_train_batch_dominance):.4f}; F1 = {(dominance_train_f1_accum / num_train_batch_dominance):.4f}\n'
        reachability_log_str = f'train epoch_{epoch_idx}: {num_train_batch_reachability} reachability batches; loss = {(reachability_train_loss_accum / num_train_batch_reachability)}; mean_t_f1 = {(reachability_trace_f1_accum / num_train_batch_liveness):.4f}; p = {(reachability_train_precision_accum / num_train_batch_reachability):.4f}; r = {(reachability_train_recall_accum / num_train_batch_reachability):.4f}; F1 = {(reachability_train_f1_accum / num_train_batch_reachability):.4f}\n'
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
        if if_log and epoch_idx % model_save_gap == 0:
            dfa_baseline_model.save_model(file_name=f'{params_hash}.epoch_{epoch_idx}')
            print('the model has been saved~')
        if iterate_entire_train_set:
            train_sampler.reset_sample_id_iter()
            train_feedback_generator = dfa_samplers.FeedbackGenerator_all_tasks(dfa_sampler=train_sampler,
                                                                                batch_size=params_dict['dfa_sampler'][
                                                                                    'batch_size'])
        sample_path_processor.dump_errored_samples_to_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Please input the params filename')
    poj104_params_savedir = ''
    parser.add_argument('--params_savedir', type=str, required=False, default=poj104_params_savedir)
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument('--clear', action='store_true')
    parser.add_argument('--unlog', action='store_true')
    parser.add_argument('--model_save_gap', type=int, default=1)
    args = parser.parse_args()
    if not os.path.isfile(os.path.join(args.params_savedir, args.params)):
        assert False, 'the specified params does not exist!'
    train(params_savedir=args.params_savedir,
          params_filename=args.params,
          if_clear=args.clear,
          if_log=not args.unlog,
          model_save_gap=args.model_save_gap)
