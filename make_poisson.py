import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import jax
import jax.numpy as jnp
import pickle
from synthetic import Poisson
import time

mus = jnp.asarray([.5, 1., 2.])
seq_num = 500

t_max = 10
key = jax.random.PRNGKey(69)

poisson = Poisson(mus)
data = list()

for i in range(seq_num):
    cur_key, key = jax.random.split(key)
    start_time = time.time()
    ts, marks = poisson.sample(t_max, cur_key)
    seq = list()
    for j in range(len(ts)):
        event = {'time_since_start': ts[j].item(), 'type_event': marks[j].item()}
        seq.append(event)
    data.append(seq)
    print(f'Sequence {i+1} processed for {time.time()-start_time} seconds.')
train_set = dict()
train_set['train'] = data[:int(.6*seq_num)]
dev_set = dict()
dev_set['dev'] = data[int(.6*seq_num):int(.8*seq_num)]
test_set = dict()
test_set['test'] = data[int(.8*seq_num):]
path = f'./my_datasets/my_poisson'
os.makedirs(path, exist_ok=True)
with open(f'{path}/train.pkl', 'wb') as f:
    pickle.dump(train_set, f)
with open(f'{path}/dev.pkl', 'wb') as f:
    pickle.dump(dev_set, f)
with open(f'{path}/test.pkl', 'wb') as f:
    pickle.dump(test_set, f)
with open(f'{path}/poisson.npy', 'wb') as f:
    np.savez(f, mus=np.asarray(mus), t_max=np.asarray(t_max))
