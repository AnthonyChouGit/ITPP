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
from eval import eval_intensity
import numpy as np
from synthetic import Hawkes
import jax.numpy as jnp

# jax.config.update("jax_debug_infs", True)
# jax.config.update("jax_debug_nans", True)


# Data loading
if args.dataname in ['hawkes']:
    train_data = load_data_hym(args.dataname, 'train', args.max_len, args.max_train_seqs)
    num_types = train_data.num_types()
    test_data = load_data_hym(args.dataname, 'test', args.max_len, -1, num_types)
    val_data = load_data_hym(args.dataname, 'dev', args.max_len, -1, num_types)
test_loader = test_data.get_dataloader(test_batch_size)

# Process horizon value for CTPP
if args.method == 'ctpp':
    channels = args.horizon.split(';')
    horizon = [c.split(',') for c in channels]
    horizon = jax.tree.map(lambda x: float(x), horizon)
else:
    horizon = None

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


model = eqx.filter_eval_shape(MTPP, args.method, args.hdim, num_types, args.hdim, init_key, {
    'num_components': args.components, 'nlayers': args.layers, 'nhead': args.nhead, 'horizon': horizon, 
    'omega': args.omega, 'siren_layers': args.siren_layers, 'num_steps': args.num_steps, 
    'reg': args.reg
})

model = eqx.tree_deserialise_leaves(f'save/{args.title}/model', model)
nll_key, predict_key, intensity_key = jax.random.split(test_key, 3)

path = './my_datasets'
with open(f'{path}/{args.dataname}/hawkes.npy', 'rb') as f:
    params = np.load(f)
    As = jnp.asarray(params['As'])
    aa = jnp.asarray(params['aa'])
    mus = jnp.asarray(params['mus'])
    t_max = params['t_max']
hawkes = Hawkes((As, aa, mus))
mape = eval_intensity(model, hawkes, test_loader, intensity_key, scale, 2000, t_max)
print(f'mape={mape}')