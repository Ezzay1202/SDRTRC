from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_PEMS
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'PEMS': Dataset_PEMS,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    # Important for SDR-TRC: validation/test predictions from different
    # models must be saved in the same deterministic sample order. Therefore
    # only the training split is shuffled.
    if flag == 'train':
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'val':
        shuffle_flag = False
        # Keep True by default to match original XLinear evaluation protocol;
        # exp_main.test() now also supports False via np.concatenate.
        drop_last = bool(getattr(args, 'eval_drop_last', 1))
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'test':
        shuffle_flag = False
        drop_last = bool(getattr(args, 'eval_drop_last', 1))
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        raise ValueError(f"Unsupported data split flag: {flag}")

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
