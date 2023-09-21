dfa_params = {
    'task_related': {'task_name': 'yzd_livenesss'},
    'dfa_net_related': {'hidden_dim': 0,
                        'if_encode_hints': True,
                        'if_decode_hints': True,
                        'if_use_lstm': True,
                        'encoder_init': 'default',
                        'dropout_prob': 0,
                        'hint_teacher_forcing': 0.8,
                        'hint_repred_mode': 'soft',
                        'nb_msg_passing_steps': 5,
                        },
    'processor_related': {'processor_kind': 'gat_dfa',
                          'nb_heads': 8,
                          'activation_name': 'relu',
                          'use_ln': True},
    'training_related': {'batch_size': 2,
                         'seed': 6},
    '_related': {'sample_id_savepath': '',
                 'sourcegraph_dir': '',
                 'errorlog_savepath': '',
                 'dataset_savedir': '',
                 'statistics_savepath': ''},
    'sample_related': {'max_iteration': 5,
                       'max_num_pp': 100,
                       'gkt_edges_rate': 1.5,
                       'selected_num_ip': 5}
}
