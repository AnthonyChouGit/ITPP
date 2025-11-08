import arg_parse
import os
args = arg_parse.parse()
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.'

import setproctitle
import jax
import equinox as eqx
import jax.numpy as jnp
import optax
import time
from jaxtyping import PyTree, Array
import numpy as np
from data import load_data_hym, shuffle_seed
from mtpp import MTPP
from tensorboardX import SummaryWriter
from eval import train_loss, eval_nll, avg_denorm_nll
from models.modules.utils import denorm_ll
import json
from math import ceil

# jax.config.update("jax_debug_infs", True)
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)

seed = 69


shuffle_seed(seed)

setproctitle.setproctitle(args.title)

# Set up the folder for model saving
save_path = f'save/{args.title}'
if not os.path.exists(save_path):
    os.mkdir(save_path)

with open(f'{save_path}/config.json', 'w') as f:
    json.dump(args.__dict__, f, indent=2)
                            
# Data loading
if args.dataname in ['stackoverflow', 'retweet', 'mimic', 'taobao', 'taxi', 'earthquake', 'hawkes', 'poisson']:
    train_data = load_data_hym(args.dataname, 'train', args.max_len, args.max_train_seqs)
    num_types = train_data.num_types()
    val_data = load_data_hym(args.dataname, 'dev', args.max_len, -1, num_types)
    test_data = load_data_hym(args.dataname, 'test', args.max_len, -1, num_types)

train_loader = train_data.get_dataloader(args.batch_size, True)
val_loader = val_data.get_dataloader(args.eval_batch_size)
test_loader = test_data.get_dataloader(args.eval_batch_size)



key = jax.random.PRNGKey(96)
init_key, train_key, val_key = jax.random.split(key, 3)

# Get the maximum sequence length
max_len = max([train_data.max_length(), val_data.max_length(), test_data.max_length()])

# Get maximum interval
dt_max = train_data.get_dtmax()
dt_mean = train_data.dt_stat()[1]

model = MTPP(args.method, num_types, args.hdim, init_key, {
    'nhead': args.nhead,
    'reg': args.reg
})

# Initialize parameters for training
base_lr = args.lr
batch_num = ceil(len(train_data) / args.batch_size )
scheduler = optax.warmup_cosine_decay_schedule(args.mlr, base_lr, args.warmups, args.max_epoch*batch_num)

optim = optax.adamw(scheduler, weight_decay=args.weight_decay)

opt_state = optim.init(eqx.filter(model, eqx.is_array))

# Set up the tensorboard
tb_writer = SummaryWriter(f'results/{args.title}', flush_secs=5)

@eqx.filter_jit
def update_model(model: eqx.Module, opt_state: PyTree, grads: PyTree):
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return model, opt_state

if args.max_dt > 0:
    scale = dt_max / args.max_dt
else:
    scale = dt_mean



best = 1e9

# Trainning loop
for epoch in range(1, args.max_epoch+1):

    # Keep track of the loss sum and event number
    ll_total = 0.
    num_events = 0
    dt_ll_total = 0.
    mark_ll_total = 0.

    # Set up a timer for the current epoch
    start_time = time.time()

    # Data loop


    for batch in train_loader:
        ts, marks, mask = batch

        cur_train_key, train_key = jax.random.split(train_key)
        # t0 = time.time()
        out, grads = eqx.filter_jit(eqx.filter_value_and_grad)(train_loss, has_aux=True)(model, ts, marks, mask, cur_train_key, scale)
        loss, aux = out # loss: scalar
        ll, dt_ll, mark_ll = aux # (N, )
        # Parameter updating
        model, opt_state = update_model(model, opt_state, grads)
        # Add up the values
        ll_total += ll.sum()
        dt_ll_total += dt_ll.sum()
        mark_ll_total += mark_ll.sum()
        num_events += mask[:, 1:].sum()
        # batch_time = time.time() - t0
        # print(f'Batch finished in {batch_time} seconds.')
    # Record the time used for the current epoch
    train_time = time.time() - start_time
    nll_avg, dt_nll_avg, mark_nll_avg = avg_denorm_nll(ll_total, dt_ll_total, mark_ll_total, num_events, scale)

    # Log the training loss on the tensorboard
    tb_writer.add_scalar('train/nll', nll_avg, epoch)
    tb_writer.add_scalar('train/t-nll', dt_nll_avg, epoch)
    tb_writer.add_scalar('train/m-nll', mark_nll_avg, epoch)
    tb_writer.flush()
    # Print the time used
    print(f'Epoch {epoch} finished in {train_time} seconds.')

    
    # Validation & Testing
    if epoch % args.val_steps == 0:
        print('evaluating...')
        val_key, test_key, cur_val_key = jax.random.split(val_key, 3)
        start_time = time.time()
        nll_per, time_nll_per, mark_nll_per = eval_nll(model, val_loader, val_key, scale)
        tb_writer.add_scalar('val/nll', np.asarray(nll_per), epoch)
        tb_writer.add_scalar('val/t-nll', np.asarray(time_nll_per), epoch)
        tb_writer.add_scalar('val/m-nll', np.asarray(mark_nll_per), epoch)
        nll_per, time_nll_per, mark_nll_per = eval_nll(model, test_loader, test_key, scale)
        tb_writer.add_scalar('test/nll', np.asarray(nll_per), epoch)
        tb_writer.add_scalar('test/t-nll', np.asarray(time_nll_per), epoch)
        tb_writer.add_scalar('test/m-nll', np.asarray(mark_nll_per), epoch)
        end_time = time.time()
        runtime = end_time - start_time
        print(f'Finished in {runtime} seconds.')
        if nll_per < best:
            print(f'Saving model to \'{save_path}/model\'...')
            eqx.tree_serialise_leaves(f'{save_path}/model', model)
            print('Model saved.')
            best = nll_per
        tb_writer.flush()


tb_writer.close()
