from jaxtyping import Array, Float, PyTree
import jax
import jax.numpy as jnp
import equinox as eqx

@eqx.filter_jit
def forward_pass(module_list: list, x: Float[Array, "hdim"]):
    for layer in module_list:
        x = layer(x)
    return x

@eqx.filter_jit
def get_dts(ts: Float[Array, "T"]):
    dts = ts[1:] - ts[:-1]
    dts = jnp.concatenate((jnp.zeros(1), dts))
    # jax.debug.print('{}', dts.min())

    dts = dts.clip(min=0.)
    return dts

def pad_to_len(mat: Array, max_len, pad_val=0.):
    pads = jnp.full((max_len-mat.shape[0], *mat.shape[1:]), pad_val, dtype=mat.dtype)
    cat = jnp.concatenate((mat, pads), 0)
    return cat

@eqx.filter_jit
def normalize(feat: Float[Array, "dim"], mean: Float[Array, "dim"], std: Float[Array, "dim"]):
    out = (feat - mean) / (std+1e-6)
    return out

@eqx.filter_jit
def denormalize(normalized: Float[Array, "dim"], mean: Float[Array, "dim"], std: Float[Array, "dim"]):
    feat = normalized * std + mean
    return feat

# When we divide the feature by a factor larger than 1, the value of the feature decreases, making 
# the distribution mass more condensed, causing ll to increase.
# In order to change the ll to its original scale, we need to minus a positive number, i.e. log(std)
@eqx.filter_jit
def denorm_ll(norm_ll: Float[Array, ""], scale: float):
    origin_ll = norm_ll - jnp.log(scale)
    return origin_ll

@eqx.filter_jit
def process_t(ts: Float[Array, "T"], scale: float):
    ts = (ts-ts[0])/scale
    return ts

@eqx.filter_jit
def process_slice(hist: tuple, pred: tuple, scale: float):
    hist_ts, hist_marks = hist
    pred_ts, pred_marks = pred
    T = hist_ts.shape[0]
    ts = jnp.concatenate((hist_ts, pred_ts)) # (T+T_, )
    ts = process_t(ts, scale)
    return (ts[:T], hist_marks), (ts[T:], pred_marks)