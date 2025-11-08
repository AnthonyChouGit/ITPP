import jax.numpy as jnp
from sklearn.metrics import root_mean_squared_error, accuracy_score, f1_score
from jaxtyping import Array, Float, Int, Bool, PyTree
import jax
import equinox as eqx
from models.modules.utils import denorm_ll, process_t, process_slice
import numpy as np
import time

@eqx.filter_jit
def train_loss(model: callable, ts: Float[Array, "N T"], marks: Int[Array, "N T"], mask: Bool[Array, "N T"], key: Array, t_scale: float=1.):
    N = ts.shape[0]
    ts = jax.vmap(process_t, (0, None))(ts, t_scale)
    key = jax.random.split(key, N)
    out = jax.vmap(model)(ts, marks, mask, key)
    ll, dt_ll, mark_ll = out[:3]
    reg = out[3] if len(out)>3 else jnp.asarray(0)
    loss = -ll.mean()+reg.mean()
    aux = (ll, dt_ll, mark_ll)
    return loss, aux

def eval_nll(model: callable, dataloader, key: Array, t_scale: float):
    model = eqx.nn.inference_mode(model)
    total_events = 0
    total_ll = 0.
    total_mark_ll = 0.
    total_time_ll = 0.
    for batch in dataloader:
        ts, marks, mask = batch
        ts = jax.vmap(process_t, (0, None))(ts, t_scale)
        cur_key, key = jax.random.split(key)
        out = jax.vmap(model)(ts, marks, mask, jax.random.split(cur_key, ts.shape[0]))
        ll, dt_ll, mark_ll = out[:3]
        total_ll += ll.sum()
        total_mark_ll += mark_ll.sum()
        total_time_ll += dt_ll.sum()
        total_events += mask[:, 1:].sum()
    nll_avg, dt_nll_avg, mark_nll_avg = avg_denorm_nll(total_ll, total_time_ll, total_mark_ll, total_events, t_scale)
    return nll_avg.item(), dt_nll_avg.item(), mark_nll_avg.item()

def eval_one_step_sampling(model: callable, dataloader, num_samples: int, dt_max: float, key: Array, t_scale: float=1.):
    model = eqx.nn.inference_mode(model)
    all_real_marks = list()
    all_pred_marks = list()
    all_real_dts = list()
    all_pred_dts = list()
    for batch in dataloader:
        ts, marks, mask = batch
        # ts, marks, mask = clip_redundant(ts, marks, mask)
        ts = jax.vmap(process_t, (0, None))(ts, t_scale)
        cur_key, key = jax.random.split(key)
        sample_tuple, real_tuple, sample_mask = jax.vmap(model.sequence_one_step_sample, (0, 0, 0, 0, None, None))(ts, marks, mask, jax.random.split(cur_key, ts.shape[0]), dt_max/t_scale, num_samples)
        sample_dts, sample_marks = sample_tuple # (N, T-1, num_samples)
        real_dts, real_marks = real_tuple # (N, T-1)
        sample_dts *= t_scale
        real_dts *= t_scale
        all_pred_dts.append(sample_dts[sample_mask])
        all_pred_marks.append(sample_marks[sample_mask])
        all_real_dts.append(real_dts[sample_mask])
        all_real_marks.append(real_marks[sample_mask])
    all_pred_dts = jnp.concatenate(all_pred_dts, 0) # (XXX, num_samples)
    all_pred_marks = jnp.concatenate(all_pred_marks, 0) # (XXX, num_samples)
    all_real_dts = jnp.concatenate(all_real_dts, 0) # (XXX, )
    all_real_marks = jnp.concatenate(all_real_marks, 0) # (XXX, )
    
    # Expand and flatten to the same shape
    all_pred_dts = all_pred_dts.flatten()
    all_pred_marks = all_pred_marks.flatten()
    all_real_dts = jnp.broadcast_to(all_real_dts[:, None], (all_real_dts.shape[0], num_samples)).flatten()
    all_real_marks = jnp.broadcast_to(all_real_marks[:, None], (all_real_marks.shape[0], num_samples)).flatten()

    rmse = root_mean_squared_error(all_real_dts, all_pred_dts)
    acc = accuracy_score(all_real_marks, all_pred_marks)
    return rmse, acc

def eval_one_step_predict(model: callable, dataloader, dt_max: float, key: Array, t_scale: float=1.):
    model = eqx.nn.inference_mode(model)
    all_real_marks = list()
    all_pred_marks = list()
    all_real_dts = list()
    all_pred_dts = list()
    for batch in dataloader:
        start = time.time()
        ts, marks, mask = batch
        ts = jax.vmap(process_t, (0, None))(ts, t_scale)
        cur_key, key = jax.random.split(key)
        predict_tuple, real_tuple, mask = jax.vmap(model.sequence_one_step_predict, (0, 0, 0, 0, None))(ts, marks, mask, jax.random.split(cur_key, ts.shape[0]), dt_max/t_scale)
        dt_predict, mark_predict = predict_tuple
        # assert jnp.all(dt_predict[mask] >= 0), "Predicted time deltas must be non-negative."
        dt_real, mark_real = real_tuple
        dt_predict = np.array(dt_predict[mask])
        dt_predict = dt_predict.clip(min=0.)
        # print(dt_predict.min())
        mark_predict = np.array(mark_predict[mask])
        dt_real = np.array(dt_real[mask])
        mark_real = np.array(mark_real[mask])
        all_pred_dts.append(dt_predict)
        all_pred_marks.append(mark_predict)
        all_real_dts.append(dt_real)
        all_real_marks.append(mark_real)
        print(f'Batch finished in {time.time() - start} seconds.')
    all_pred_dts = np.concatenate(all_pred_dts, 0)
    all_pred_marks = np.concatenate(all_pred_marks, 0)
    all_real_dts = np.concatenate(all_real_dts, 0)
    all_real_marks = np.concatenate(all_real_marks, 0)
    all_pred_dts *= t_scale
    all_real_dts *= t_scale
    # print(all_pred_dts.max(), all_pred_dts.min())
    # print(all_real_dts.max(), all_real_dts.min())
    rmse = root_mean_squared_error(all_real_dts, all_pred_dts)
    acc = f1_score(all_real_marks, all_pred_marks, average='weighted')
    return rmse, acc

def eval_intensity(model: callable, tpp, dataloader, key: Array, t_scale: float, num_samples: int, t_max: float):
    model = eqx.nn.inference_mode(model)
    ape_sum = 0.
    total_sample_num = 0
    for batch in dataloader:
        start = time.time()
        ts, marks, mask = batch # (N, T)

        ts = jnp.asarray(ts)
        marks = jnp.asarray(marks)
        mask = jnp.asarray(mask)

        # Fill padded times with the boundary value
        ts = jnp.where(mask, ts, t_max)

        # Append a boundary value at the last
        ts = jnp.concatenate((ts, jnp.full((ts.shape[0], 1), t_max)), -1) # (N, T+1)

        # Shift and rescale the data for models
        ts_ = jax.vmap(process_t, (0, None))(ts, t_scale) # (N, T+1)

        # Time intervals after each event
        dts_ = ts_[:, 1:] - ts_[:, 1:] # (N, T)

        T = ts_.shape[1]
        enc_key, intensity_key, key = jax.random.split(key, 3)

        # Encode the processed ts_ to obtain the states at each step
        states = jax.vmap(model.encode)(ts_[:, :-1], marks, mask, jax.random.split(enc_key, marks.shape[0])) # (N, T, ...)

        # For each state
        for i in range(T):
            
            # Get the i-th state
            state_cur = jax.tree.map(lambda x: x[:, i], states) # (N, ...)

            # Uniformly generate samples in [0, 1] 
            temp_samples = jnp.linspace(0., 1., num_samples) # (num_samples, )

            # Rescale the samples according to the i-th dt_
            dt_samples_ = temp_samples[None, :] * dts_[:, i, None] # (N, num_samples)

            cur_intensity_key,  intensity_key = jax.random.split(intensity_key)

            # Get the intensities values at each sampled interval with respect to state_cur
            pred_intensities = jax.vmap(model.intensities_at)(state_cur, dt_samples_, jax.random.split(cur_intensity_key, dt_samples_.shape[0])) # (N, num_samples, num_types)

            # Map the intensities back to the orginal scale
            pred_intensities = pred_intensities / t_scale # (num_samples, num_types)

            # Map the sampled intervals to the original scale
            dt_samples = dt_samples_ * t_scale

            mask_cur = mask.at[i+1:].set(False)

            # Get the real intensities from the original point process TODO: modify this line to just changint the mask with jitting
            real_intensities = jax.vmap(jax.vmap(eqx.filter_jit(tpp.intensities_at), (None, None, None, 0)))(ts[:, :-1], marks, mask_cur, dt_samples) # (N, num_samples, num_types)

            # Compute the absolute percentage error for all samples at this step
            ape = jnp.abs(pred_intensities-real_intensities) / real_intensities # (N, num_samples, num_types)

            # Clear up the ape values for padded positions
            ape = jnp.where(mask[:, i, None, None], ape, 0)

            # Compute the number of valid interval samples
            valid_samples = num_samples * mask[:, i, None, None].sum()

            # Add up the ape sum
            ape_sum += ape.sum((0, 1)) # (num_types, )

            # Add up the sample num
            total_sample_num += valid_samples

            # print(f'i={i} finished.')
        print(f'Batch finished in {time.time() - start} seconds.')

    # Compute MAPE for each event type
    mape = ape_sum / total_sample_num # (num_types, )

    # Compute the average of mape across types
    mape = mape.mean()

    return mape
            

@eqx.filter_jit
def avg_denorm_nll(ll_total, dt_ll_total, mark_ll_total, num_events, t_scale):
    nll_avg = -denorm_ll(ll_total / num_events, t_scale)
    dt_nll_avg = -denorm_ll(dt_ll_total / num_events, t_scale)
    mark_nll_avg = -mark_ll_total / num_events
    return nll_avg, dt_nll_avg, mark_nll_avg
