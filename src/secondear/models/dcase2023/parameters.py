# Vendored from https://github.com/sharathadavanne/seld-dcase2023 (parameters.py)
# commit 24d14c9ebe6878062dc3b68eb07dd0100722f530
# MIT License, (c) Sharath Adavanne / DCASE 2023 baseline authors
# Modified for second-ear: removed print/exit calls (unknown id raises ValueError). See README.md in this folder.
# ruff: noqa
# fmt: off
# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.


def get_params(argv='1'):
    # ########### default parameters ##############
    params = dict(
        quick_test=True,     # To do quick test. Trains/test on small subset of dataset, and # of epochs
    
        finetune_mode = False,  # Finetune on existing model, requires the pretrained model path set - pretrained_model_weights
        pretrained_model_weights='models/1_1_foa_dev_split6_model.h5',

        # INPUT PATH
        # dataset_dir='DCASE2020_SELD_dataset/',  # Base folder containing the foa/mic and metadata folders
        dataset_dir='/scratch/asignal/partha/DCASE2023/DCASE2023_SELD_dataset',

        # OUTPUT PATHS
        # feat_label_dir='DCASE2020_SELD_dataset/feat_label_hnet/',  # Directory to dump extracted features and labels
        feat_label_dir='/scratch/asignal/partha/DCASE2023/DCASE2023_SELD_dataset/seld_feat_label',
 
        model_dir='models/',            # Dumps the trained models and training curves in this folder
        dcase_output_dir='results/',    # recording-wise results are dumped in this path.

        # DATASET LOADING PARAMETERS
        mode='dev',         # 'dev' - development or 'eval' - evaluation dataset
        dataset='foa',       # 'foa' - ambisonic or 'mic' - microphone signals

        #FEATURE PARAMS
        fs=24000,
        hop_len_s=0.02,
        label_hop_len_s=0.1,
        max_audio_len_s=60,
        nb_mel_bins=64,

        use_salsalite = False, # Used for MIC dataset only. If true use salsalite features, else use GCC features
        fmin_doa_salsalite = 50,
        fmax_doa_salsalite = 2000,
        fmax_spectra_salsalite = 9000,

        # MODEL TYPE
        multi_accdoa=False,  # False - Single-ACCDOA or True - Multi-ACCDOA
        thresh_unify=15,    # Required for Multi-ACCDOA only. Threshold of unification for inference in degrees.

        # DNN MODEL PARAMETERS
        label_sequence_length=50,    # Feature sequence length
        batch_size=128,              # Batch size
        dropout_rate=0.05,           # Dropout rate, constant for all layers
        nb_cnn2d_filt=64,           # Number of CNN nodes, constant for each layer
        f_pool_size=[4, 4, 2],      # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer

        self_attn=True,
        nb_heads=8,
        nb_self_attn_layers=2,
        
        nb_rnn_layers=2,
        rnn_size=128,

        nb_fnn_layers=1,
        fnn_size=128,             # FNN contents, length of list = number of layers, list value = number of nodes

        nb_epochs=100,              # Train for maximum epochs
        lr=1e-3,

        # METRIC
        average='macro',        # Supports 'micro': sample-wise average and 'macro': class-wise average
        lad_doa_thresh=20
    )

    # ########### User defined parameters ##############
    if argv == '1':
        pass

    elif argv == '2':
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = False

    elif argv == '3':
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True

    elif argv == '4':
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False

    elif argv == '5':
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = False

    elif argv == '6':
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True

    elif argv == '7':
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = True

    elif argv == '999':
        params['quick_test'] = True

    else:
        raise ValueError('unknown argument {}'.format(argv))

    feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution
    params['t_pool_size'] = [feature_label_resolution, 1, 1]     # CNN time pooling
    params['patience'] = int(params['nb_epochs'])     # Stop training if patience is reached

    if '2020' in params['dataset_dir']:
        params['unique_classes'] = 14 
    elif '2021' in params['dataset_dir']:
        params['unique_classes'] = 12
    elif '2022' in params['dataset_dir']:
        params['unique_classes'] = 13
    elif '2023' in params['dataset_dir']:
        params['unique_classes'] = 13


    return params
