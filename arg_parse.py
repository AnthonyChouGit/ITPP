import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--val_steps', type=int, default=10)
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--dataname', type=str, default='stackoverflow')
    parser.add_argument('--hdim', type=int, default=64)
    parser.add_argument('--title', type=str, default='test')
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmups', type=int, default=0)
    parser.add_argument('--method', type=str, default='itpp')
    parser.add_argument('--nhead', type=int, default=2)
    parser.add_argument('--max_len', type=int, default=200)
    parser.add_argument('--reg', type=float, default=0)
    parser.add_argument('--mlr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_train_seqs', type=int, default=-1)
    parser.add_argument('--max_dt', type=float, default=-1)
    args = parser.parse_args()
    assert args.warmups < args.max_epoch
    
    return args

def load_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--title', type=str, default='test')
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--cuda', type=str, default='0')
    args = parser.parse_args()
    return args