import json
from arg_parse import load_parse
args = load_parse()
test_batch_size = args.test_batch_size
import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.'
with open(f'save/{args.title}/config.json', 'r') as f:
    args.__dict__ = json.load(f)
from data import load_data_hym
import jax
from mtpp import MTPP
import equinox as eqx
from eval import eval_nll, eval_one_step_predict

# jax.config.update("jax_debug_infs", True)
# jax.config.update("jax_debug_nans", True)


# Data loading
if args.dataname in ['stackoverflow', 'retweet', 'mimic', 'taobao', 'taxi', 'earthquake', 'poisson', 'hawkes']:
    train_data = load_data_hym(args.dataname, 'train', args.max_len, args.max_train_seqs)
    num_types = train_data.num_types()
    test_data = load_data_hym(args.dataname, 'test', args.max_len, -1, num_types)
    val_data = load_data_hym(args.dataname, 'dev', args.max_len, -1, num_types)
test_loader = test_data.get_dataloader(test_batch_size)


key = jax.random.PRNGKey(69)
init_key, test_key = jax.random.split(key)

# Get the maximum sequence length
max_len = test_data.max_length()

dt_max = train_data.get_dtmax()
dt_mean = train_data.dt_stat()[1]

temp = test_data.dt_stat()[-1]
if args.max_dt > 0:
    scale = dt_max / args.max_dt
else:
    scale = dt_mean


model = eqx.filter_eval_shape(MTPP, args.method, num_types, args.hdim, init_key, {
    'nhead': args.nhead,
    'reg': args.reg
})

model = eqx.tree_deserialise_leaves(f'save/{args.title}/model', model)
nll_key, predict_key = jax.random.split(test_key)
nll_per, time_nll_per, mark_nll_per = eval_nll(model, test_loader, nll_key, scale)
print(f'NLL: {nll_per}, T-NLL: {time_nll_per}, M-NLL: {mark_nll_per}.')

rmse, acc = eval_one_step_predict(model, test_loader, dt_max, predict_key, scale)
print(f'RMSE: {rmse}, F1: {acc}.')
