import os.path
import pickle, json
import argparse
import numpy as np
from typing import Optional
from clrs._src import dfa_utils, dfa_samplers, dfa_processors, dfa_baselines

import haiku as hk
import jax


def filter_sample_list(statistics_savepath,
                       errored_sample_ids,
                       max_num_pp,
                       sample_id_savepath):
    with open(statistics_savepath) as statistics_loader:
        statistics_dict = json.load(statistics_loader)

    sample_id_list = []
    with open(sample_id_savepath) as sample_ids_loader:
        for line in sample_ids_loader.readlines():
            sample_id = line.strip()
            if sample_id in statistics_dict:
                if statistics_dict[sample_id] > max_num_pp:
                    continue
            else:
                assert sample_id in errored_sample_ids
                continue
            sample_id_list.append(sample_id)
    return sample_id_list


def test(test_info_savedir: str,
         test_info_filename: str,
         statistics_filepath: str,
         ckpt_idx: Optional[int],
         if_clear, if_log):
    test_info_hash, test_info_savepath = dfa_utils.rename_params_file(params_savedir=test_info_savedir,
                                                                      params_filename=test_info_filename,
                                                                      if_test_params=True)
    with open(test_info_savepath) as test_info_loader:
        test_info_dict = json.load(test_info_loader)
    trained_model_params_id = test_info_dict['trained_model_info']['trained_model_params_id']
    print(f'We are testing with {test_info_hash}... \n(the trained model is: {trained_model_params_id})')
    trained_model_params_savedir = test_info_dict['trained_model_info']['trained_model_params_savedir']
    trained_model_params_savepath = os.path.join(trained_model_params_savedir, f'{trained_model_params_id}.params')
    trained_model_params_dict = dfa_utils.parse_params(params_hash=trained_model_params_id,
                                                       params_filepath=trained_model_params_savepath,
                                                       statistics_filepath=statistics_filepath,
                                                       for_model_test=True)
    sample_path_processor = dfa_utils.SamplePathProcessor(**trained_model_params_dict['sample_path_processor'])
    rng = np.random.RandomState(seed=test_info_dict['random_seed'])
    test_info_dict['test_sample_loader']['dfa_version'] = trained_model_params_dict['train_sample_loader'][
        'dfa_version']
    test_sample_loader = dfa_utils.SampleLoader(sample_path_processor=sample_path_processor,
                                                seed=rng.randint(2 ** 32),
                                                **test_info_dict['test_sample_loader'])
    test_sampler = dfa_samplers.DFASampler(task_name=trained_model_params_dict['task']['task_name'],
                                           sample_id_list=filter_sample_list(statistics_savepath=statistics_filepath,
                                                                             errored_sample_ids=sample_path_processor.errored_sample_ids,
                                                                             max_num_pp=
                                                                             test_info_dict['test_sample_loader'][
                                                                                 'max_num_pp'],
                                                                             sample_id_savepath=
                                                                             test_info_dict['dfa_sampler'][
                                                                                 'test_sample_id_savepath']),
                                           seed=rng.randint(2 ** 32),
                                           sample_loader=test_sample_loader)
    test_feedback_generator = dfa_samplers.FeedbackGenerator(dfa_sampler=test_sampler,
                                                             batch_size=test_info_dict['dfa_sampler']['batch_size'])
    processor_factory = dfa_processors.get_dfa_processor_factory(**trained_model_params_dict['processor'])
    dfa_baseline_model = dfa_baselines.DFABaselineModel(processor_factory=processor_factory,
                                                        **trained_model_params_dict['dfa_net'],
                                                        **trained_model_params_dict['baseline_model'])
    _, init_feedback = next(test_feedback_generator)
    dfa_baseline_model.init(features=init_feedback.features,
                            seed=test_info_dict['random_seed'] + 1)
    ckpt_savedir = trained_model_params_dict['baseline_model']['checkpoint_path']
    ckpt_filename_list = []
    if ckpt_idx is None:
        #   test all the ckpts
        ckpt_count = 0
        for fn in os.listdir(ckpt_savedir):
            if not fn.startswith('.'):
                ckpt_count += 1
        for idx in range(ckpt_count):
            ckpt_filename_list.append(f'{trained_model_params_id}.epoch_{idx + 1}')
    else:
        ckpt_filename_list.append(f'{trained_model_params_id}.epoch_{ckpt_idx}')

    rng_key = jax.random.PRNGKey(rng.randint(2 ** 32))
    test_log_savepath = os.path.join(test_info_dict['test_log_savedir'], f'{test_info_hash}.test_log')
    if os.path.isfile(test_log_savepath) and if_clear and if_log:
        os.system(f'rm {test_log_savepath}')
        print('old log has been removed!')
    log_str = ''
    for ckpt_filename in ckpt_filename_list:
        ckpt_idx = ckpt_filename.split('_')[-1].strip()
        print(f'ckpt_{ckpt_idx} on testing...')
        # ckpt_savepath = os.path.join(ckpt_savedir, ckpt_filename)
        # with open(ckpt_savepath, 'rb') as f:
        #     restored_model_params_state = pickle.load(f)
        print(
            f'dfa_test line 96, dfa_baseline_model.ckpt_path = {dfa_baseline_model.checkpoint_path}; ckpt_file_name = {ckpt_filename}')
        # dfa_baseline_model.params = hk.data_structures.merge(dfa_baseline_model.params,
        #                                                      restored_model_params_state['params'])
        # dfa_baseline_model.opt_state = restored_model_params_state['opt_state']
        dfa_baseline_model.restore_model(file_name=ckpt_filename)
        test_batch_idx, test_precision_accum, test_recall_accum, test_f1_accum = 0.0, 0.0, 0.0, 0.0
        num_test_batch_liveness, liveness_test_precision_accum, liveness_test_recall_accum, liveness_test_f1_accum = 0.0, 0.0, 0.0, 0.0
        num_test_batch_dominance, dominance_test_precision_accum, dominance_test_recall_accum, dominance_test_f1_accum = 0.0, 0.0, 0.0, 0.0
        num_test_batch_reachability, reachability_test_precision_accum, reachability_test_recall_accum, reachability_test_f1_accum = 0.0, 0.0, 0.0, 0.0
        # pos_num_accum, total_num_accum = 0.0, 0.0
        while test_batch_idx < test_info_dict['num_steps_per_ckpt']:
            task_name_this_batch, test_feedback_batch = next(test_feedback_generator)
            new_rng_key, rng_key = jax.random.split(rng_key)
            test_precision, test_recall, test_f1, _, _ = dfa_baseline_model.get_measures(
                rng_key=new_rng_key,
                feedback=test_feedback_batch)
            if task_name_this_batch is not None:
                new_log_str = f'test with ckpt_{ckpt_idx} batch_{int(test_batch_idx)} ({task_name_this_batch}): precision = {test_precision}; recall = {test_recall}; F1 = {test_f1}\n'
                if task_name_this_batch == 'liveness':
                    liveness_test_precision_accum += test_precision
                    liveness_test_recall_accum += test_recall
                    liveness_test_f1_accum += test_f1
                    num_test_batch_liveness += 1
                elif task_name_this_batch == 'dominance':
                    dominance_test_precision_accum += test_precision
                    dominance_test_recall_accum += test_recall
                    dominance_test_f1_accum += test_f1
                    num_test_batch_dominance += 1
                elif task_name_this_batch == 'reachability':
                    reachability_test_precision_accum += test_precision
                    reachability_test_recall_accum += test_recall
                    reachability_test_f1_accum += test_f1
                    num_test_batch_reachability += 1
                else:
                    print('dfa_test line 138, unrecognized task_name!!')
                    assert False
            else:
                new_log_str = f'test with ckpt_{ckpt_idx} batch_{int(test_batch_idx)}: precision = {test_precision}; recall = {test_recall}; F1 = {test_f1}\n'
                test_precision_accum += test_precision
                test_recall_accum += test_recall
                test_f1_accum += test_f1
            test_batch_idx += 1
            print(new_log_str, end='')
            log_str += new_log_str
            # print(f'positive_num = {positive_num}; total_num = {total_num}; pos_percentage = {float(positive_num) / float(total_num)}')
            # pos_num_accum += positive_num
            # total_num_accum += total_num
        if num_test_batch_liveness + num_test_batch_dominance + num_test_batch_reachability > 0:
            liveness_log_str = f'test with ckpt_{ckpt_idx}: {num_test_batch_liveness} liveness batches; precision = {liveness_test_precision_accum / num_test_batch_liveness}; recall = {liveness_test_recall_accum / num_test_batch_liveness}; F1 = {liveness_test_f1_accum / num_test_batch_liveness}\n'
            dominance_log_str = f'test with ckpt_{ckpt_idx}: {num_test_batch_dominance} dominance batches; precision = {dominance_test_precision_accum / num_test_batch_dominance}; recall = {dominance_test_recall_accum / num_test_batch_dominance}; F1 = {dominance_test_f1_accum / num_test_batch_dominance}\n'
            reachability_log_str = f'test with ckpt_{ckpt_idx}: {num_test_batch_reachability} reachability batches; precision = {reachability_test_precision_accum / num_test_batch_reachability}; recall = {reachability_test_recall_accum / num_test_batch_reachability}; F1 = {reachability_test_f1_accum / num_test_batch_reachability}\n'
            new_log_str = liveness_log_str + reachability_log_str + dominance_log_str
        else:
            new_log_str = f'test with ckpt_{ckpt_idx}: precision = {test_precision_accum / test_batch_idx}; recall = {test_recall_accum / test_batch_idx}; F1 = {test_f1_accum / test_batch_idx}\n'
        print(new_log_str, end='')
        log_str += new_log_str
        # print(f'positive_num = {pos_num_accum / test_batch_idx}; total_num = {total_num_accum / test_batch_idx}; pos_percentage = {pos_num_accum / total_num_accum}')
        if if_log:
            with open(test_log_savepath, 'a') as log_recorder:
                log_recorder.write(log_str)
        del log_str
        log_str = ''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Please input the params filename')
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument('--ckpt_idx', type=int, default=None, required=False)
    parser.add_argument('--clear', action='store_true')
    parser.add_argument('--unlog', action='store_true')
    args = parser.parse_args()
    test_info_savedir = '/data_hdd/lx20/yzd_workspace/Params/TestInfoPOJ104/'
    statistics_filepath = '/data_hdd/lx20/yzd_workspace/Datasets/Statistics/POJ104Statistics/poj104_statistics.json'
    if not os.path.isfile(os.path.join(test_info_savedir, args.params)):
        print('the specified params does not exist!')
        exit(134)
    test(test_info_savedir=test_info_savedir,
         test_info_filename=args.params,
         statistics_filepath=statistics_filepath,
         ckpt_idx=args.ckpt_idx,
         if_clear=args.clear,
         if_log=not args.unlog)