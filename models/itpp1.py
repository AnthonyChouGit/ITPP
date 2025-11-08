import equinox as eqx
from jaxtyping import Array, Float, Int, Bool, PyTree
import jax.numpy as jnp
import jax
from .modules.utils import forward_pass, get_dts
from .modules.ode import integrate, integrate2, integrate_save
import math

def scaled_dot_product_attention(q: Float[Array, "T_q key_dim"], k: Float[Array, "T_k key_dim"], 
                v: Float[Array, "T_k val_dim"], mask: Bool[Array, "T_q T_k"]):
    temperature = q.shape[-1] ** 0.5
    attn = jnp.matmul(q / temperature, k.T) # (T_q, T_k)
    attn = jnp.where(mask, attn, -1e9)
    attn = jax.nn.softmax(attn, -1)
    output = jnp.matmul(attn, v)
    return output

class AdaptiveAttentionLayer(eqx.Module):
    W: eqx.nn.Linear
    b: Float[Array, "K nhead*3*hdim"]
    nhead: int
    out_proj: eqx.nn.Linear
    norm: eqx.nn.LayerNorm

    def __init__(self, hdim: int, K: int, nhead: int, key: Array):
        lim = 1 / math.sqrt(hdim)
        key1, key2, key3 = jax.random.split(key, 3)
        self.out_proj = eqx.nn.Linear(nhead * hdim, hdim, key=key3)
        # self.W = jax.random.uniform(key1, (K, hdim, nhead * 3 * hdim), minval=-lim, maxval=lim)
        self.W = eqx.nn.Linear(hdim, nhead * 3 * hdim, False, key=key1)
        self.b = jax.random.uniform(key2, (K, nhead * 3 * hdim), minval=-lim, maxval=lim)
        self.nhead = nhead
        self.norm = eqx.nn.LayerNorm((K, hdim), 1e-8)
    
    def __call__(self, Z: Float[Array, "K hdim"]):
        K, hdim = Z.shape
        # q_k_v = jax.vmap(linear_forward)(self.W, self.b, Z).reshape(K, self.nhead, 3*hdim)  # (K, nhead*3*hdim)
        q_k_v = (jax.vmap(self.W)(Z) + self.b).reshape(K, self.nhead, 3*hdim) # (K, nhead, 3*hdim)
        q, k, v = jnp.split(q_k_v, 3, axis=-1)  # (K, nhead, hdim), (K, nhead, hdim), (K, nhead, hdim)
        q = jnp.transpose(q, (1, 0, 2))  # (nhead, K, hdim)
        k = jnp.transpose(k, (1, 0, 2)) # (nhead, K, hdim)
        v = jnp.transpose(v, (1, 0, 2)) # (nhead, K, hdim)
        output = jax.vmap(scaled_dot_product_attention, (0, 0, 0, None))(
            q, k, v, jnp.ones((K, K), dtype=bool)
        ) # (nhead, K, hdim)
        output = jnp.transpose(output, (1, 0, 2)) # (K, nhead, hdim)
        output = output.reshape(K, -1)  # (K, nhead*hdim)
        output = jax.vmap(self.out_proj)(output)
        output = self.norm(output+Z)

        return output
    
class GRUNet_noinput(eqx.Module):
    rz_net: list
    g_net: list

    def __init__(self, hdim: int, key: Array):
        key1, key2 = jax.random.split(key)
        self.rz_net = [
            eqx.nn.Linear(hdim, 2*hdim, key=key1),
            jax.nn.sigmoid    
        ]
        self.g_net = [
            eqx.nn.Linear(hdim, hdim, key=key2),
            jax.nn.tanh
        ]

    def __call__(self, h: Float[Array, "hdim"]):
        rz = forward_pass(self.rz_net, h)
        r, z = jnp.split(rz, 2, -1)
        g = forward_pass(self.g_net, r*h)
        return z, g

class Func(eqx.Module):
    gru: GRUNet_noinput
    intensity_fn: list
    attn: AdaptiveAttentionLayer

    def __init__(self, hdim: int, num_types: int, nhead: int, key: Array):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        self.gru = GRUNet_noinput(hdim, key1)
        self.intensity_fn = [
            eqx.nn.Linear(hdim, 2*hdim, key=key2),
            jax.nn.silu,
            eqx.nn.Linear(2*hdim, 1, key=key4),
            jax.nn.softplus
        ]
        self.attn = AdaptiveAttentionLayer(hdim, num_types, nhead,  key3)

    def get_intensities(self, Z: Float[Array, "K hdim"]):
        K = Z.shape[0]
        H = self.attn(Z)
        intensities = jax.vmap(forward_pass, (None, 0))(self.intensity_fn, H).squeeze(-1) # (K, )
        return intensities
    
    def __call__(self, t: Float[Array, ""], state: PyTree, args=None):
        energy, Lambda, Z = state
        z, g = jax.vmap(self.gru)(Z)  # (K, hdim), (K, hdim)
        dZ = jax.nn.tanh((1-z)*(g-Z))
        intensities = self.get_intensities(Z)
        intensity = intensities.sum()
        d_energy = ((dZ**2).sum()) / (dZ.size)
        return d_energy, intensity, dZ
    
    def get_H(self, Z: Float[Array, "K hdim"]):
        K = Z.shape[0]
        H = self.attn(Z) # (K, hdim)
        return H

class Jump(eqx.Module):
    gru: GRUNet_noinput

    def __init__(self, hdim: int, key: Array):
        self.gru = GRUNet_noinput(hdim, key)

    def __call__(self, mark: Int[Array, ""], Z: Float[Array, "K hdim"]):
        z, g = self.gru(Z[mark])
        post_jump_mark = z * Z[mark] + (1-z) * g
        Z = Z.at[mark].set(post_jump_mark)
        return Z

class Encoder(eqx.Module):
    func: Func
    jump: Jump
    hdim: int
    num_types: int
    init_state: Array

    def __init__(self, hdim: int, num_types: int, nhead: int, key: Array):
        key1, key2, key3 = jax.random.split(key, 3)
        self.func = Func(hdim, num_types, nhead, key1)
        self.jump = Jump(hdim, key2)
        self.hdim = hdim
        self.num_types = num_types
        lim = 1 / math.sqrt(hdim)
        self.init_state = jax.random.uniform(key3, (num_types, hdim), minval=-lim, maxval=lim)

    def step(self, state0: PyTree, dt: Float[Array, ""], mark: Int[Array, ""]):
        state1 = integrate(self.func, 0., dt, state0, None)
        energy1, Lambda1, Z1_prior = state1
        Z1_post = self.jump(mark, Z1_prior)
        return energy1, Lambda1, Z1_post, Z1_prior
    
    def extrapolate(self, state0: PyTree, dt: Float[Array, ""]):
        state1 = integrate(self.func, 0., dt, state0, None)
        return state1
    
    def extrapolate_multiple(self, state0: PyTree, dts: Float[Array, "num_samples"]):
        state1 = integrate_save(self.func, 0., dts, state0, None) # (num_samples, ...)
        return state1
    
    def __call__(self, dts: Float[Array, "T"], marks: Int[Array, "T"]):
        state0 = (jnp.asarray(0.), jnp.asarray(0.), self.init_state)  # (energy, Lambda, Z)
        inputs = (dts, marks)
        def step(h, x):
            dt, mark = x
            energy1, Lambda1, Z1_post, Z1_prior = self.step(h, dt, mark)
            carry = (energy1, Lambda1, Z1_post)
            y = (Z1_post, Z1_prior)
            return carry, y
        carry, ys = jax.lax.scan(step, state0, inputs)
        energy, Lambda, Z_end = carry
        Zs_post, Zs_prior = ys
        return Zs_post, Zs_prior, energy, Lambda, Z_end

class ITPP(eqx.Module):
    enc: Encoder
    energy_reg: float

    def __init__(self, hdim: int, num_types: int, nhead: int, energy_reg: float, key: Array):
        self.enc = Encoder(hdim, num_types, nhead, key)
        self.energy_reg = energy_reg

    def __process_Z(self, Z: Float[Array, "K hdim"], mark: Int[Array, ""]):
        intensities = self.enc.func.get_intensities(Z)
        intensity = intensities[mark]
        log_intensity = jnp.log(intensity + 1e-8)
        mark_prob = intensity / (intensities.sum()+1e-8)
        mark_ll = jnp.log(mark_prob+1e-8)
        return log_intensity, mark_ll

    @eqx.filter_jit
    def __call__(self, ts: Float[Array, "T"], marks: Int[Array, "T"], mask: Bool[Array, "T"], key=None):
        dts = get_dts(ts)
        dts = jnp.where(mask, dts, 0.) # (T)
        Zs_post, Zs_prior, energy, Lambda, Z_end = self.enc(dts, marks)
        log_intensity, mark_ll = jax.vmap(self.__process_Z)(Zs_prior[1:], marks[1:])
        mark_ll = jnp.where(mask[1:], mark_ll, 0.)  # (T-1)
        log_intensity = jnp.where(mask[1:], log_intensity, 0.)
        mark_ll = mark_ll.sum()
        ll = log_intensity.sum() - Lambda
        time_ll = ll - mark_ll
        return ll, time_ll, mark_ll, energy* self.energy_reg

    def _predict(self, Z0: Float[Array, "K hdim"], dt_max: Float[Array, ""]):
        energy0 = jnp.asarray(0.)
        Lambda0 = jnp.asarray(0.)
        state0 = (energy0, Lambda0, Z0)
        int_tf0 = jnp.asarray(0.)
        H0 = (state0, int_tf0)
        def func(t, H, args=None):
            state, int_tf = H
            energy, Lambda, Z = state
            d_state = self.enc.func(t, state)
            d_energy, intensity, dZ = d_state

            f_dt = intensity * jnp.exp(-Lambda)
            tf = t * f_dt
            return d_state, tf
        _, Efdt = integrate2(func, 0., jnp.asarray(dt_max), H0, None)
        Efdt = Efdt.clip(min=0.)
        state1 = self.enc.extrapolate(state0, Efdt)
        energy1, Lambda1, Z1 = state1
        intensities = self.enc.func.get_intensities(Z1)
        mark_predict = intensities.argmax(-1)
        return Efdt, mark_predict
    
    @eqx.filter_jit
    def rolling_predict(self, ts: Float[Array, "T"], marks: Int[Array, "T"], mask: Bool[Array, "T"], dt_max: float, key=None):
        dts = get_dts(ts)
        dts = jnp.where(mask, dts, 0.).clip(min=0.) # (T)
        Zs_post, Zs_prior, energy, Lambda, Z_end = self.enc(dts, marks)
        dt_predict, mark_predict = jax.vmap(self._predict, (0, None))(Zs_post[:-1], dt_max)  # (T-1, ), (T-1, )
        # jax.lax.cond(jnp.all(dt_predict>=0), true_fn, false_fn, dt_predict)
        return (dt_predict, mark_predict), (dts[1:], marks[1:]), mask[1:]
    
    @eqx.filter_jit
    def encode(self, ts: Float[Array, "T"], marks: Int[Array, "T"], mask: Bool[Array, "T"], key=None):
        dts = get_dts(ts)
        dts = jnp.where(mask, dts, 0.) # (T)
        Zs_post, Zs_prior, energy, Lambda, Z_end = self.enc(dts, marks)
        return jnp.zeros((Zs_post.shape[0], )), jnp.zeros((Zs_post.shape[0], )), Zs_post
    
    @eqx.filter_jit
    def intensities_at(self, state: PyTree, dts: Float[Array, "num_samples"], key=None):
        state1 = self.enc.extrapolate_multiple(state, dts) # (num_samples, )
        _, _, Z1 = state1 # Z1: (num_samples, K, hdim)
        intensities = jax.vmap(self.enc.func.get_intensities)(Z1) # (num_samples, num_types)
        return intensities
