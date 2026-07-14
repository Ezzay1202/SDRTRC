import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2025, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Informer, Transformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


    # DLinear
    #parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

    # PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # Formers 
    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # XLinear
    parser.add_argument('--c_ff', default=1, type=int)
    parser.add_argument('--t_ff', default=1, type=int)
    parser.add_argument('--c_dropout', default=0., type=float)
    parser.add_argument('--t_dropout', default=0., type=float)
    parser.add_argument('--embed_dropout', type=float, default=0.1)
    parser.add_argument('--usenorm', default=1, type=int)


    # HSDNet / SCDNet options
    parser.add_argument('--hsd_state_dim', type=int, default=32, help='HSDNet state embedding dimension')
    parser.add_argument('--hsd_dep_dim', type=int, default=16, help='HSDNet variable dependency embedding dimension')
    parser.add_argument('--hsd_num_groups', type=int, default=4, help='HSDNet number of horizon groups')
    parser.add_argument('--hsd_topk', type=int, default=0, help='HSDNet top-k source variables for dependency attention; 0 means dense soft attention')
    parser.add_argument('--hsd_gate_init', type=float, default=-3.0, help='HSDNet initial dependency gate logit')
    parser.add_argument('--hsd_dropout', type=float, default=0.1, help='HSDNet dropout in state/dependency branch')
    parser.add_argument('--hsd_temperature', type=float, default=1.0, help='HSDNet dependency attention temperature')
    parser.add_argument('--hsd_fusion', type=str, default='residual', choices=['interp', 'interpolate', 'residual', 'res'], help='HSDNet fusion type')
    parser.add_argument('--hsd_disable_dep', type=int, default=0, help='hard-disable dependency branch; do not instantiate it')
    parser.add_argument('--hsd_zero_init', type=int, default=1, help='zero-initialize dependency residual output layer')
    parser.add_argument('--hsd_rng_safe_init', type=int, default=1, help='restore RNG state after dependency branch initialization')
    parser.add_argument('--hsd_residual_scale', type=float, default=0.05, help='maximum normalized magnitude of bounded dependency correction')
    parser.add_argument('--hsd_aux_lambda', type=float, default=0.001, help='regularization weight for final dependency correction magnitude')
    parser.add_argument('--hsd_risk_use_quantile', type=int, default=1, help='use batch quantile as risk threshold')
    parser.add_argument('--hsd_risk_quantile', type=float, default=0.70, help='risk threshold quantile; higher means fewer corrected samples')
    parser.add_argument('--hsd_risk_threshold', type=float, default=1.0, help='absolute risk threshold when quantile threshold is disabled')
    parser.add_argument('--hsd_risk_sharpness', type=float, default=8.0, help='sharpness of risk sigmoid')
    parser.add_argument('--hsd_risk_floor', type=float, default=0.0, help='minimum risk multiplier')
    parser.add_argument('--hsd_detach_risk', type=int, default=1, help='detach non-parametric risk score from gradient')
    parser.add_argument('--hsd_debug', type=int, default=0, help='print one-batch HSDNet debug statistics')
    parser.add_argument('--hsd_residual_target_lambda', type=float, default=0.05, help='HSDNet-v3 weight for residual-target supervision')
    parser.add_argument('--hsd_residual_target_clip', type=float, default=3.0, help='clip normalized residual target for stable residual supervision')
    parser.add_argument('--hsd_residual_target_risk_weight', type=int, default=1, help='weight residual-target loss by risk score')
    parser.add_argument('--hsd_residual_target_min_weight', type=float, default=0.10, help='minimum weight for residual-target loss when risk weighting is enabled')


    # SDR-TR options (clean naming; hsd_* arguments are still accepted as legacy aliases inside the model)
    parser.add_argument('--sdr_state_dim', type=int, default=128, help='SDR-TR state embedding dimension')
    parser.add_argument('--sdr_dep_dim', type=int, default=64, help='SDR-TR dependency embedding dimension')
    parser.add_argument('--sdr_num_groups', type=int, default=4, help='number of horizon groups for dependency residuals')
    parser.add_argument('--sdr_topk', type=int, default=0, help='top-k source variables for dependency attention; 0 means dense attention')
    parser.add_argument('--sdr_dropout', type=float, default=0.05, help='dropout in state/dependency branch')
    parser.add_argument('--sdr_temperature', type=float, default=1.0, help='dependency attention temperature')
    parser.add_argument('--sdr_disable_dep', type=int, default=0, help='hard-disable residual dependency branch; should reproduce the base XLinear branch')
    parser.add_argument('--sdr_version', type=int, default=2, help='SDRTR version: 2=attention branch (default), 3=MLP branch')
    parser.add_argument('--sdr_zero_init', type=int, default=1, help='zero-initialize raw residual output layer')
    parser.add_argument('--sdr_rng_safe_init', type=int, default=1, help='restore RNG state after residual branch initialization')
    parser.add_argument('--sdr_residual_scale', type=float, default=0.15, help='bound for tanh residual intervention in normalized space')
    parser.add_argument('--sdr_aux_lambda', type=float, default=0.001, help='regularization weight for final trust-region correction magnitude')
    parser.add_argument('--sdr_trust_logit', type=float, default=None, help='fixed trust-region radius logit; None = use branch default (v2: -2.0, v3: -0.8)')
    parser.add_argument('--sdr_trust_radius', type=float, default=-1.0, help='direct fixed trust-region radius; <=0 means use sigmoid(sdr_trust_logit)')
    parser.add_argument('--sdr_use_learnable_gate', type=int, default=1, help='use learnable horizon gate for trust-region radius (0=fixed, 1=learnable)')
    parser.add_argument('--sdr_risk_use_quantile', type=int, default=1, help='use batch quantile as state-risk threshold')
    parser.add_argument('--sdr_risk_quantile', type=float, default=0.70, help='risk threshold quantile')
    parser.add_argument('--sdr_risk_threshold', type=float, default=1.0, help='absolute risk threshold when quantile threshold is disabled')
    parser.add_argument('--sdr_risk_sharpness', type=float, default=8.0, help='sharpness of risk sigmoid')
    parser.add_argument('--sdr_risk_floor', type=float, default=0.0, help='minimum risk multiplier')
    parser.add_argument('--sdr_detach_risk', type=int, default=1, help='detach non-parametric risk score from gradient')
    parser.add_argument('--sdr_debug', type=int, default=0, help='print SDR-TR debug statistics')
    parser.add_argument('--sdr_residual_target_lambda', type=float, default=0.05, help='weight for residual-target supervision')
    parser.add_argument('--sdr_residual_target_clip', type=float, default=3.0, help='clip normalized residual target for stable residual supervision')
    parser.add_argument('--sdr_residual_target_risk_weight', type=int, default=1, help='weight residual-target loss by state risk')
    parser.add_argument('--sdr_residual_target_min_weight', type=float, default=0.10, help='minimum residual-target loss weight under risk weighting')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
    parser.add_argument('--save_val_pred', type=int, default=1,
                        help='save validation predictions after training; needed for residual calibration')
    parser.add_argument('--eval_split', type=str, default='both', choices=['train', 'val', 'test', 'both'],
                        help='which split to evaluate when is_training=0; both exports val and test')
    parser.add_argument('--eval_drop_last', type=int, default=1,
                        help='drop incomplete final batch for val/test. Keep 1 to match original XLinear protocol; set 0 for full split export')

    args = parser.parse_args()

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)


    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, flag='test')
            if args.save_val_pred:
                print('>>>>>>>saving validation predictions : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting, flag='val')

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                    args.model,
                                                                                                    args.data,
                                                                                                    args.features,
                                                                                                    args.seq_len,
                                                                                                    args.label_len,
                                                                                                    args.pred_len,
                                                                                                    args.d_model,
                                                                                                    args.n_heads,
                                                                                                    args.e_layers,
                                                                                                    args.d_layers,
                                                                                                    args.d_ff,
                                                                                                    args.factor,
                                                                                                    args.embed,
                                                                                                    args.distil,
                                                                                                    args.des, ii)

        exp = Exp(args)  # set experiments
        if args.eval_split == 'both':
            print('>>>>>>>export validation : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1, flag='val')
            print('>>>>>>>export test : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1, flag='test')
        else:
            print('>>>>>>>export {} : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(args.eval_split, setting))
            exp.test(setting, test=1, flag=args.eval_split)
        torch.cuda.empty_cache()
        