import json

import jax
import argparse
# import hashlib
import numpy as np
import os
from clrs._src import dfa_samplers
from clrs._src import dfa_utils
from clrs._src import dfa_processors
from clrs._src import dfa_baselines
from clrs._src import dfa_train


def rename_vali_params_file(params_savedir,
                            params_filename):
    cur_params_filepath = os.path.join(params_savedir, params_filename)
    params_hash = dfa_utils.compute_hash(file_path=cur_params_filepath)
    with open(os.path.join(params_savedir, params_filename)) as vali_params_loader:
        train_params_id = json.load(vali_params_loader)['trained_model_info']['train_params_id']
    new_params_filename = f'{train_params_id}.{params_hash}.vali'
    tmp_list = params_filename.split('.')
    if len(tmp_list) == 3:
        if params_filename == new_params_filename:
            return params_hash, cur_params_filepath
    new_params_filepath = os.path.join(params_savedir, new_params_filename)
    os.system(f'mv {cur_params_filepath} {new_params_filepath}')
    if not os.path.isfile(new_params_filepath):
        print('where is the renamed params file???')
        exit(1)
    return params_hash, new_params_filepath


def validate(vali_params_savedir: str,
             vali_params_filename: str,
             ckpt_idx_list,
             ckpt_step: int,
             if_clear,
             if_log):
    vali_params_id, vali_params_savepath = rename_vali_params_file(
        params_savedir=vali_params_savedir,
        params_filename=vali_params_filename)
    with open(vali_params_savepath) as vali_params_loader:
        vali_params_dict = json.load(vali_params_loader)
    assert vali_params_dict['dfa_sampler']['batch_size'] == 1, 'Sorry but we only support batch_size = 1 by now'
    train_params_id = vali_params_dict['trained_model_info']['train_params_id']
    print(f'We are testing with {vali_params_id}... \n(the trained model is: {train_params_id})')
    train_params_savedir = vali_params_dict['trained_model_info']['train_params_savedir']
    train_params_savepath = os.path.join(train_params_savedir, f'{train_params_id}.train')
    train_params_dict = dfa_train.parse_train_params(params_hash=train_params_id,
                                                     params_filepath=train_params_savepath)
    sample_path_processor = dfa_utils.SamplePathProcessor(
        sourcegraph_dir=vali_params_dict['sample_path_processor']['sourcegraph_dir'],
        errorlog_savepath=vali_params_dict['sample_path_processor']['errorlog_savepath'])
    rng = np.random.RandomState(seed=vali_params_dict['random_seed'])
    vali_params_dict['vali_sample_loader']['dfa_version'] = train_params_dict['train_sample_loader']['dfa_version']
    vali_sample_loader = dfa_utils.SampleLoader(sample_path_processor=sample_path_processor,
                                                seed=rng.randint(2 ** 32),
                                                **vali_params_dict['vali_sample_loader'])
    filtered_vali_sample_ids = dfa_utils.filter_sample_list(
        full_statistics_savepath=vali_params_dict['vali_filepath']['full_statistics_savepath'],
        errored_sample_ids=sample_path_processor.errored_sample_ids,
        max_num_pp=vali_params_dict['vali_sample_loader']['max_num_pp'],
        min_num_pp=vali_params_dict['vali_sample_loader']['min_num_pp'],
        cfg_edges_rate=vali_params_dict['vali_sample_loader']['cfg_edges_rate'],
        sample_id_savepath=vali_params_dict['vali_filepath']['vali_sample_id_savepath'])
    iterate_entire_dataset = True if vali_params_dict['num_steps_per_ckpt'] < 0 else False
    vali_sampler = dfa_samplers.DFASampler(task_name=train_params_dict['task']['task_name'],
                                           sample_id_list=filtered_vali_sample_ids,
                                           seed=rng.randint(2 ** 32),
                                           sample_loader=vali_sample_loader,
                                           iterate_all=iterate_entire_dataset)

    vali_feedback_generator = dfa_samplers.FeedbackGenerator_all_tasks(dfa_sampler=vali_sampler,
                                                                       batch_size=vali_params_dict['dfa_sampler'][
                                                                           'batch_size'])
    processor_factory = dfa_processors.get_dfa_processor_factory(**train_params_dict['processor'])
    ckpt_savedir = train_params_dict['baseline_model']['checkpoint_path']
    # print(f'dfa_vali line 78 ckpt_savedir = {ckpt_savedir}')
    # exit(666)
    del train_params_dict['baseline_model']['checkpoint_path']
    del train_params_dict['dfa_net']['encode_hints']
    del train_params_dict['dfa_net']['decode_hints']
    if 'just_one_layer' in train_params_dict['dfa_net'] and train_params_dict['dfa_net']['just_one_layer'] and \
            vali_params_dict['vali_sample_loader']['expected_trace_len'] == 2:
        train_params_dict['dfa_net']['just_one_layer'] = True
    else:
        train_params_dict['dfa_net']['just_one_layer'] = False
    dfa_baseline_model = dfa_baselines.DFABaselineModel(processor_factory=processor_factory,
                                                        **train_params_dict['dfa_net'],
                                                        **train_params_dict['baseline_model'],
                                                        **vali_params_dict['dfa_net'],
                                                        checkpoint_path=ckpt_savedir)
    # if iterate_entire_dataset:
    #     vali_sampler.reset_sample_id_iter()
    _, _, init_feedback = next(vali_feedback_generator)
    next(vali_feedback_generator)
    next(vali_feedback_generator)
    dfa_baseline_model.init(features=init_feedback.features,
                            seed=vali_params_dict['random_seed'] + 1)
    vali_log_savedir = vali_params_dict['vali_filepath']['vali_log_savedir']
    vali_log_savedir = os.path.join(vali_log_savedir, f'{train_params_id}.{vali_params_id}_vali_log')
    os.makedirs(vali_log_savedir, exist_ok=True)
    vali_log_savepath_list = []
    ckpt_filename_list = []
    ckpt_count_in_total = 0
    for fn in os.listdir(ckpt_savedir):
        if not fn.startswith('.'):
            ckpt_count_in_total += 1
    if ckpt_idx_list is not None:
        assert ckpt_step is None
        for ckpt_idx in ckpt_idx_list:
            ckpt_filename_list.append(f'{train_params_id}.epoch_{ckpt_idx}')
            vali_log_savepath_list.append(os.path.join(vali_log_savedir, f'epoch_{ckpt_idx}.vali_log'))
    elif ckpt_step is not None:
        for idx in range(ckpt_count_in_total):
            if idx % ckpt_step == 0:
                ckpt_filename_list.append(f'{train_params_id}.epoch_{idx}')
                vali_log_savepath_list.append(os.path.join(vali_log_savedir, f'epoch_{idx}.vali_log'))
    else:
        for idx in range(ckpt_count_in_total):
            ckpt_filename_list.append(f'{train_params_id}.epoch_{idx}')
            vali_log_savepath_list.append(os.path.join(vali_log_savedir, f'epoch_{idx}.vali_log'))
    # else:
    #     for ckpt_idx in ckpt_idx_list:
    #         ckpt_filename_list.append(f'{train_params_id}.epoch_{ckpt_idx}')
    #         vali_log_savepath_list.append(os.path.join(vali_log_savedir, f'epoch_{ckpt_idx}.vali_log'))
    rng_key = jax.random.PRNGKey(rng.randint(2 ** 32))
    log_str = ''
    for ckpt_filename, vali_log_savepath in zip(ckpt_filename_list, vali_log_savepath_list):
        if os.path.isfile(vali_log_savepath) and if_clear and if_log:
            os.system(f'rm {vali_log_savepath}')
            print('old log has been removed!')
        if os.path.isfile(vali_log_savepath) and not if_clear and if_log:
            print(f'Previous vali log with {vali_log_savepath} exists! so we just skip!')
            continue
            # exit(1)
        ckpt_idx = ckpt_filename.split('_')[-1].strip()
        print(f'ckpt_{ckpt_idx} on validation...')
        dfa_baseline_model.restore_model(file_name=ckpt_filename)
        vali_batch_idx = 0.0
        num_vali_batch_liveness, liveness_vali_precision_accum, liveness_vali_recall_accum, liveness_vali_f1_accum, liveness_vali_trace_f1_accum = 0.0, 0.0, 0.0, 0.0, 0.0
        num_vali_batch_dominance, dominance_vali_precision_accum, dominance_vali_recall_accum, dominance_vali_f1_accum, dominance_vali_trace_f1_accum = 0.0, 0.0, 0.0, 0.0, 0.0
        num_vali_batch_reachability, reachability_vali_precision_accum, reachability_vali_recall_accum, reachability_vali_f1_accum, reachability_vali_trace_f1_accum = 0.0, 0.0, 0.0, 0.0, 0.0
        sampled_id_remain = ''
        liveness_done, reachability_done, dominance_done = False, False, False
        while True:
            if iterate_entire_dataset:
                try:
                    sampled_ids_this_batch, task_name_this_batch, vali_feedback_batch = next(vali_feedback_generator)
                except:  # Stop
                    break
            else:
                if vali_batch_idx == vali_params_dict[
                    'num_steps_per_ckpt'] and liveness_done and reachability_done and dominance_done:
                    break
                sampled_ids_this_batch, task_name_this_batch, vali_feedback_batch = next(vali_feedback_generator)
            cur_sampled_id = sampled_ids_this_batch[0]
            if not cur_sampled_id == sampled_id_remain:
                new_log_str = f'\ntest with ckpt_{ckpt_idx}_batch_{int(vali_batch_idx)} {cur_sampled_id}: '
                print(new_log_str)
                log_str += new_log_str
                sampled_id_remain = cur_sampled_id
                vali_batch_idx += 1
                liveness_done, reachability_done, dominance_done = False, False, False
            new_rng_key, rng_key = jax.random.split(rng_key)
            mean_trace_f1, vali_precision, vali_recall, vali_f1 = dfa_baseline_model.get_measures(
                rng_key=new_rng_key,
                feedback=vali_feedback_batch,
                return_hints=True,
                print_full_trace_h_f1_list=not if_log)
            full_trace_len_this_batch = vali_feedback_batch.features.mask_dict['full_trace_len']
            print(f'full_trace_len = {full_trace_len_this_batch}. (dfa_vali line 163)')
            # exit(666)
            new_log_str = f'{task_name_this_batch}: mean_t_f1 = {mean_trace_f1:.4f}; p = {vali_precision:.4f}; r = {vali_recall:.4f}; f1 = {vali_f1:.4f}. '
            print(new_log_str)
            log_str += new_log_str
            if task_name_this_batch == 'liveness':
                liveness_vali_trace_f1_accum += mean_trace_f1
                liveness_vali_precision_accum += vali_precision
                liveness_vali_recall_accum += vali_recall
                liveness_vali_f1_accum += vali_f1
                num_vali_batch_liveness += 1
                liveness_done = True
            elif task_name_this_batch == 'reachability':
                reachability_vali_trace_f1_accum += mean_trace_f1
                reachability_vali_precision_accum += vali_precision
                reachability_vali_recall_accum += vali_recall
                reachability_vali_f1_accum += vali_f1
                num_vali_batch_reachability += 1
                reachability_done = True
            elif task_name_this_batch == 'dominance':
                dominance_vali_trace_f1_accum += mean_trace_f1
                dominance_vali_precision_accum += vali_precision
                dominance_vali_recall_accum += vali_recall
                dominance_vali_f1_accum += vali_f1
                num_vali_batch_dominance += 1
                dominance_done = True
            else:
                print('dfa_train line 150, unrecognized task_name!!')
                assert False
            if if_log and vali_batch_idx % 500 == 0:
                with open(vali_log_savepath, 'a') as log_recorder:
                    log_recorder.write(log_str)
                del log_str
                log_str = ''
        liveness_log_str = f'\n vali on ckpt_{ckpt_idx}: {int(num_vali_batch_liveness)} liveness batches; mean_t_f1 = {(liveness_vali_trace_f1_accum / num_vali_batch_liveness):.4f}; p = {(liveness_vali_precision_accum / num_vali_batch_liveness):.4f}; r = {(liveness_vali_recall_accum / num_vali_batch_liveness):.4f}; F1 = {(liveness_vali_f1_accum / num_vali_batch_liveness):.4f}\n'
        dominance_log_str = f'vali on ckpt_{ckpt_idx}: {int(num_vali_batch_dominance)} dominance batches; mean_t_f1 = {(dominance_vali_trace_f1_accum / num_vali_batch_dominance):.4f}; p = {(dominance_vali_precision_accum / num_vali_batch_dominance):.4f}; r = {(dominance_vali_recall_accum / num_vali_batch_dominance):.4f}; F1 = {(dominance_vali_f1_accum / num_vali_batch_dominance):.4f}\n'
        reachability_log_str = f'vali on ckpt_{ckpt_idx}: {int(num_vali_batch_reachability)} reachability batches; mean_t_f1 = {(reachability_vali_trace_f1_accum / num_vali_batch_reachability):.4f}; p = {(reachability_vali_precision_accum / num_vali_batch_reachability):.4f}; r = {(reachability_vali_recall_accum / num_vali_batch_reachability):.4f}; F1 = {(reachability_vali_f1_accum / num_vali_batch_reachability):.4f}\n'
        new_log_str = liveness_log_str + reachability_log_str + dominance_log_str
        print(new_log_str, end='')
        log_str += new_log_str
        if if_log:
            with open(vali_log_savepath, 'a') as log_recorder:
                log_recorder.write(log_str)
        del log_str
        log_str = ''
        if iterate_entire_dataset:
            vali_sampler.reset_sample_id_iter()
            vali_feedback_generator = dfa_samplers.FeedbackGenerator_all_tasks(dfa_sampler=vali_sampler,
                                                                               batch_size=
                                                                               vali_params_dict['dfa_sampler'][
                                                                                   'batch_size'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Please input the params filename')
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument('--ckpt_idx', type=int, nargs="+", default=None, required=False)
    parser.add_argument('--ckpt_step', type=int, default=None)
    parser.add_argument('--clear', action='store_true')
    parser.add_argument('--unlog', action='store_true')
    args = parser.parse_args()
    params_savedir = '/data_hdd/lx20/yzd_workspace/Params/ValiParams/poj104_ValiParams'
    # statistics_filepath = '/data_hdd/lx20/yzd_workspace/Datasets/Statistics/poj104_Statistics/poj104_num_pp_statistics.json'
    if not os.path.isfile(os.path.join(params_savedir, args.params)):
        print('the specified params does not exist!')
        exit(176)
    # params_filename = 'params_not_hint_as_outpt.json'
    validate(vali_params_savedir=params_savedir,
             vali_params_filename=args.params,
             ckpt_idx_list=args.ckpt_idx,
             ckpt_step=args.ckpt_step,
             if_clear=args.clear,
             if_log=not args.unlog)
