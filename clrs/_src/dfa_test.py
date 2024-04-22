import os.path
import json
import argparse
import numpy as np
from typing import List
from clrs._src import dfa_train, dfa_utils
from clrs._src import dfa_samplers, dfa_processors, dfa_baselines

import haiku as hk
import jax


def rename_test_params_file(params_savedir,
                            params_filename):
    cur_params_filepath = os.path.join(params_savedir, params_filename)
    params_hash = dfa_utils.compute_hash(file_path=cur_params_filepath)
    with open(os.path.join(params_savedir, params_filename)) as test_params_loader:
        train_params_id = json.load(test_params_loader)['trained_model_info']['train_params_id']
    new_params_filename = f'{train_params_id}.{params_hash}.test'
    tmp_list = params_filename.split('.')
    if len(tmp_list) == 3:
        assert params_filename == new_params_filename
        return params_hash, cur_params_filepath
    new_params_filepath = os.path.join(params_savedir, new_params_filename)
    os.system(f'mv {cur_params_filepath} {new_params_filepath}')
    if not os.path.isfile(new_params_filepath):
        print('where is the renamed params file???')
        exit(1)
    return params_hash, new_params_filepath


def test(util_path_processer: dfa_utils.UtilPathProcessor,
         test_dataset_name: str,
         test_info_filename: str,
         ckpt_idx_list: List[int],
         if_clear, if_log, check_trace):
    test_info_hash, test_info_savepath = rename_test_params_file(
        params_savedir=util_path_processer.test_info_savedir(dataset_name=test_dataset_name),
        params_filename=test_info_filename)
    with open(test_info_savepath) as test_info_loader:
        test_info_dict = json.load(test_info_loader)
    # utils_path_processer = dfa_utils.UtilPathProcessor()
    assert test_info_dict['dfa_sampler']['batch_size'] == 1, 'Sorry but we only support batch_size = 1 by now'
    trained_dataset_name = test_info_dict['trained_model_info']['trained_dataset_name']
    assert test_dataset_name == test_info_dict['trained_model_info']['test_dataset_name']
    trained_model_params_id = test_info_dict['trained_model_info']['trained_model_params_id']
    print(f'We are testing with {test_info_hash}... \n(the trained model is: {trained_model_params_id})')
    # trained_model_params_savedir = test_info_dict['trained_model_info']['trained_model_params_savedir']
    with open(util_path_processer.full_statistics_filepath(test_dataset_name)) as full_statistics_loader:
        full_statistics_dict = json.load(full_statistics_loader)
    trained_model_params_savedir = util_path_processer.trained_model_params_savedir(dataset_name=trained_dataset_name)
    trained_model_params_savepath = os.path.join(trained_model_params_savedir, f'{trained_model_params_id}.params')
    trained_model_params_dict = dfa_train.parse_train_params(params_hash=trained_model_params_id,
                                                             params_filepath=trained_model_params_savepath,
                                                             statistics_filepath=util_path_processer.num_pp_statistics_filepath(
                                                                 dataset_name=trained_dataset_name))
    sample_path_processor = dfa_utils.SamplePathProcessor(
        sourcegraph_dir=trained_model_params_dict['sample_path_processor']['sourcegraph_dir'],
        errorlog_savepath=util_path_processer.errorlog_savepath(test_dataset_name))
    rng = np.random.RandomState(seed=test_info_dict['random_seed'])
    test_info_dict['test_sample_loader']['dfa_version'] = trained_model_params_dict['train_sample_loader'][
        'dfa_version']
    test_sample_loader = dfa_utils.SampleLoader(sample_path_processor=sample_path_processor,
                                                seed=rng.randint(2 ** 32),
                                                **test_info_dict['test_sample_loader'])
    iterate_entire_dataset = True if test_info_dict['num_steps_per_ckpt'] < 0 else False
    num_pp_statistics_filepath = util_path_processor.num_pp_statistics_filepath(dataset_name=test_dataset_name)
    test_sampler = dfa_samplers.DFASampler(task_name=trained_model_params_dict['task']['task_name'],
                                           sample_id_list=dfa_utils.filter_sample_list(
                                               statistics_savepath=num_pp_statistics_filepath,
                                               errored_sample_ids=sample_path_processor.errored_sample_ids,
                                               max_num_pp=
                                               test_info_dict['test_sample_loader'][
                                                   'max_num_pp'],
                                               min_num_pp=
                                               test_info_dict['test_sample_loader'][
                                                   'min_num_pp'],
                                               # sample_id_savepath=test_info_dict['dfa_sampler']['test_sample_id_savepath']
                                               sample_id_savepath=util_path_processer.test_sample_ids_savepath(
                                                   dataset_name=test_dataset_name)
                                           ),
                                           seed=rng.randint(2 ** 32),
                                           sample_loader=test_sample_loader,
                                           iterate_all=iterate_entire_dataset)
    test_feedback_generator = dfa_samplers.FeedbackGenerator_all_tasks(dfa_sampler=test_sampler,
                                                                       batch_size=test_info_dict['dfa_sampler'][
                                                                           'batch_size'])
    processor_factory = dfa_processors.get_dfa_processor_factory(**trained_model_params_dict['processor'])
    del trained_model_params_dict['baseline_model']['checkpoint_path']
    ckpt_savedir = util_path_processer.ckpt_savedir(dataset_name=trained_dataset_name,
                                                    params_hash=trained_model_params_id)
    dfa_baseline_model = dfa_baselines.DFABaselineModel(processor_factory=processor_factory,
                                                        **trained_model_params_dict['dfa_net'],
                                                        **trained_model_params_dict['baseline_model'],
                                                        checkpoint_path=ckpt_savedir)
    if iterate_entire_dataset:
        test_sampler.reset_sample_id_iter()
    _, _, init_feedback = next(test_feedback_generator)
    dfa_baseline_model.init(features=init_feedback.features,
                            seed=test_info_dict['random_seed'] + 1)
    # ckpt_savedir = trained_model_params_dict['baseline_model']['checkpoint_path']

    # test_log_savedir = os.path.join(test_info_dict['test_log_savedir'], f'{trained_model_params_id}_trained',
    #                                 f'{test_info_hash}_test')
    test_log_savedir = util_path_processer.test_log_savedir(dataset_name=test_dataset_name,
                                                            trained_model_params_id=trained_model_params_id,
                                                            test_info_hash=test_info_hash)
    os.makedirs(test_log_savedir, exist_ok=True)
    test_log_savepath_list = []
    ckpt_filename_list = []
    if ckpt_idx_list is None:
        #   test all the ckpts
        ckpt_count = 0
        for fn in os.listdir(ckpt_savedir):
            if not fn.startswith('.'):
                ckpt_count += 1
        for idx in range(ckpt_count):
            ckpt_filename_list.append(f'{trained_model_params_id}.epoch_{idx}')
            test_log_savepath_list.append(os.path.join(test_log_savedir, f'epoch_{idx}.test_log'))
    else:
        for ckpt_idx in ckpt_idx_list:
            ckpt_filename_list.append(f'{trained_model_params_id}.epoch_{ckpt_idx}')
            test_log_savepath_list.append(os.path.join(test_log_savedir, f'epoch_{ckpt_idx}.test_log'))

    rng_key = jax.random.PRNGKey(rng.randint(2 ** 32))
    # test_log_savepath = os.path.join(test_info_dict['test_log_savedir'], f'{test_info_hash}.test_log')
    # if os.path.isfile(test_log_savepath) and if_clear and if_log:
    #     os.system(f'rm {test_log_savepath}')
    #     print('old log has been removed!')
    log_str = ''
    for ckpt_filename, test_log_savepath in zip(ckpt_filename_list, test_log_savepath_list):
        ckpt_idx = ckpt_filename.split('_')[-1].strip()
        print(f'ckpt_{ckpt_idx} on testing...')
        # ckpt_savepath = os.path.join(ckpt_savedir, ckpt_filename)
        # with open(ckpt_savepath, 'rb') as f:
        #     restored_model_params_state = pickle.load(f)
        # print(f'dfa_test line 96, dfa_baseline_model.ckpt_path = {dfa_baseline_model.checkpoint_path}; ckpt_file_name = {ckpt_filename}')
        # dfa_baseline_model.params = hk.data_structures.merge(dfa_baseline_model.params,
        #                                                      restored_model_params_state['params'])
        # dfa_baseline_model.opt_state = restored_model_params_state['opt_state']
        dfa_baseline_model.restore_model(file_name=ckpt_filename)
        test_batch_idx, test_precision_accum, test_recall_accum, test_f1_accum = 0.0, 0.0, 0.0, 0.0
        num_test_batch_liveness, liveness_test_precision_accum, liveness_test_recall_accum, liveness_test_f1_accum = 0.0, 0.0, 0.0, 0.0
        num_test_batch_dominance, dominance_test_precision_accum, dominance_test_recall_accum, dominance_test_f1_accum = 0.0, 0.0, 0.0, 0.0
        num_test_batch_reachability, reachability_test_precision_accum, reachability_test_recall_accum, reachability_test_f1_accum = 0.0, 0.0, 0.0, 0.0
        # pos_num_accum, total_num_accum = 0.0, 0.0
        sampled_id_remain = ''
        case_analysis_savepath = util_path_processer.case_analysis_savepath(dataset_name=test_dataset_name,
                                                                            test_info_id=test_info_hash,
                                                                            ckpt_idx=ckpt_idx)
        case_analysis_dict = dict()
        while True:
            if iterate_entire_dataset:
                try:
                    sampled_ids_this_batch, task_name_this_batch, test_feedback_batch = next(test_feedback_generator)
                except RuntimeError:  # Stop
                    break
            else:
                if test_batch_idx == test_info_dict['num_steps_per_ckpt']:
                    break
                sampled_ids_this_batch, task_name_this_batch, test_feedback_batch = next(test_feedback_generator)
            # while test_batch_idx < test_info_dict['num_steps_per_ckpt']:
            #     task_name_this_batch, test_feedback_batch = next(test_feedback_generator)
            # assert len(sampled_ids_this_batch) == 1
            cur_sampled_id = sampled_ids_this_batch[0]
            if not cur_sampled_id == sampled_id_remain:
                sampled_id_remain = cur_sampled_id
                full_statistics_this_sample = full_statistics_dict[cur_sampled_id]
                num_pp, num_cfg, max_out_degree, max_in_degree, tl_l, tl_r, tl_d = full_statistics_this_sample
                statistics_str = f'test with ckpt_{ckpt_idx} batch_{int(test_batch_idx)}: {cur_sampled_id}\nnum_pp = {num_pp}, num_cfg = {num_cfg}, max_out_degree = {max_out_degree}, max_in_degree = {max_in_degree}; tl_l = {tl_l}; tl_r = {tl_r}; tl_d = {tl_d}'
                print(statistics_str)
                test_batch_idx += 1
                case_analysis_dict[cur_sampled_id] = [999, 999, 999]
            full_trace_len_this_batch = test_feedback_batch.features.mask_dict['full_trace_len']
            new_rng_key, rng_key = jax.random.split(rng_key)
            test_precision, test_recall, test_f1 = dfa_baseline_model.get_measures(
                rng_key=new_rng_key,
                feedback=test_feedback_batch,
                return_hints=check_trace)
            if task_name_this_batch is not None:
                # statistics_str = f'test with ckpt_{ckpt_idx} batch_{int(test_batch_idx)} {cur_sampled_id}:\nfull_trace_len = {full_trace_len_this_batch}; num_pp = {num_pp}, num_cfg = {num_cfg}, en_ratio = {en_ratio}, max_out_degree_liveness = {max_out_degree}, max_in_degree = {max_in_degree}'
                # print(statistics_str)
                new_log_str = f'{task_name_this_batch}: trace_len = {full_trace_len_this_batch}; precision = {test_precision}; recall = {test_recall}; F1 = {test_f1}\n'
                # new_log_str = f'test with ckpt_{ckpt_idx} batch_{int(test_batch_idx)} ({task_name_this_batch}): full_trace_len = {full_trace_len_this_batch}; precision = {test_precision}; recall = {test_recall}; F1 = {test_f1}\n'
                if task_name_this_batch == 'liveness':
                    liveness_test_precision_accum += test_precision
                    liveness_test_recall_accum += test_recall
                    liveness_test_f1_accum += test_f1
                    num_test_batch_liveness += 1
                    case_analysis_dict[cur_sampled_id][0] = test_f1
                elif task_name_this_batch == 'reachability':
                    reachability_test_precision_accum += test_precision
                    reachability_test_recall_accum += test_recall
                    reachability_test_f1_accum += test_f1
                    num_test_batch_reachability += 1
                    case_analysis_dict[cur_sampled_id][1] = test_f1
                elif task_name_this_batch == 'dominance':
                    dominance_test_precision_accum += test_precision
                    dominance_test_recall_accum += test_recall
                    dominance_test_f1_accum += test_f1
                    num_test_batch_dominance += 1
                    case_analysis_dict[cur_sampled_id][2] = test_f1
                else:
                    print('dfa_test line 138, unrecognized task_name!!')
                    assert False
            else:
                new_log_str = f'test with ckpt_{ckpt_idx} batch_{int(test_batch_idx)}: precision = {test_precision}; recall = {test_recall}; F1 = {test_f1}\n'
                test_precision_accum += test_precision
                test_recall_accum += test_recall
                test_f1_accum += test_f1
            # test_batch_idx += 1
            print(new_log_str, end='')
            log_str += new_log_str
            # print(f'positive_num = {positive_num}; total_num = {total_num}; pos_percentage = {float(positive_num) / float(total_num)}')
            # pos_num_accum += positive_num
            # total_num_accum += total_num
            if test_batch_idx % 500 == 0:
                with open(case_analysis_savepath, 'w') as analysis_logger:
                    json.dump(case_analysis_dict, analysis_logger, indent=3)
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
        with open(case_analysis_savepath, 'w') as analysis_logger:
            json.dump(case_analysis_dict, analysis_logger, indent=3)
        if if_log:
            if if_clear:
                with open(test_log_savepath, 'w') as log_recorder:
                    log_recorder.write(log_str)
            else:
                if os.path.isfile(test_log_savepath):
                    print(f'{test_log_savepath} has already existed! please specify if it should be cleared!')
                    exit(174)
        del log_str
        log_str = ''
        if iterate_entire_dataset:
            test_sampler.reset_sample_id_iter()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Please input the params filename')
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument('--ckpt_idx', type=int, nargs="+", default=None, required=False)
    parser.add_argument('--clear', action='store_true')
    parser.add_argument('--unlog', action='store_true')
    parser.add_argument('--check_trace', action='store_true')
    parser.add_argument('--test_dataset_name', type=str, required=True)
    args = parser.parse_args()
    assert args.test_dataset_name in ['poj104', 'tensorflow', 'linux', 'opencv', 'opencl', 'npd']
    util_path_processor = dfa_utils.UtilPathProcessor()
    # test_info_savedir = '/data_hdd/lx20/yzd_workspace/Params/TestInfoPOJ104/'
    # statistics_filepath = '/data_hdd/lx20/yzd_workspace/Datasets/Statistics/POJ104Statistics/poj104_statistics.json'
    test_info_savedir = util_path_processor.test_info_savedir(dataset_name=args.test_dataset_name)
    # statistics_filepath = util_path_processor.statistics_filepath(dataset_name=args.test_dataset_name)
    if not os.path.isfile(os.path.join(test_info_savedir, args.params)):
        print(f'dfa_test line 227, test_info_savepath = {os.path.join(test_info_savedir, args.params)}')
        print('the specified params does not exist!')
        exit(134)
    test(util_path_processer=util_path_processor,
         test_dataset_name=args.test_dataset_name,
         # test_info_savedir=test_info_savedir,
         test_info_filename=args.params,
         ckpt_idx_list=args.ckpt_idx,
         if_clear=args.clear,
         if_log=not args.unlog,
         check_trace=args.check_trace)
