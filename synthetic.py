import jax
import equinox as eqx
from jaxtyping import Float, Bool, Array, PyTree, Int
import numpy as np
import jax.numpy as jnp

class TPP(eqx.Module):
    def sample(self, t_max: float, key, **kwargs):
        raise NotImplementedError()
    
    def intensities_at(self, ts: Float[Array, "T"], marks: Int[Array, "T"], mask: Bool[Array, "T"], dt: Float[Array, ""]):
        raise NotImplementedError()

class Poisson(TPP):
    params: Array

    def __init__(self, params: Float[Array, "num_types"]):
        super().__init__()
        assert np.all(params>0)
        self.params = params

    def intensities_at(self, ts: Float[Array, "T"], marks: Int[Array, "T"], mask: Bool[Array, "T"], dt: Float[Array, ""]): # GPU function
        return self.params
    
    def sample(self, t_max: float, key, **kwargs): # CPU function
        ts = list()
        marks = list()
        num_types = len(self.params)
        for i in range(num_types):
            t = 0
            while True:
                cur_key, key = jax.random.split(key)
                exp = jax.random.exponential(cur_key).item()
                dt = exp / self.params[i]
                t += dt
                if t>= t_max:
                    break
                ts.append(t)
                marks.append(i)
        ts = np.asarray(ts)
        marks = np.asarray(marks)
        inds = np.argsort(ts)
        ts = ts[inds]
        marks = marks[inds]
        return ts, marks
    
class Hawkes(TPP):
    params: PyTree

    def __init__(self, params: PyTree):
        super().__init__()
        A, Alpha, mu = params
        assert np.all(A>0)
        assert np.all(Alpha>0)
        assert np.all(mu>0)
        self.params = params

    def intensities_at(self, ts: Float[Array, "T"], marks: Int[Array, "T"], mask: Bool[Array, "T"], dt: Float[Array, ""]):
        dts = ts[1:] - ts[:-1] # (T-1, )
        dts = jnp.where(mask[1:], dts, 0.)
        dts_reverse = jnp.flip(dts) # (T-1, )
        dist_reverse = dts_reverse.cumsum(-1) # (T-1, )
        dist = jnp.flip(dist_reverse) # (T-1, )
        dist = jnp.concatenate((dist, jnp.asarray([0.])), -1) # (T, )

        dist = dist + dt
        A, Alpha, mu = self.params # (num_types, num_types) (num_types, num_types) (num_types, )
        a = A[marks] # (T, num_types)
        alpha = Alpha[marks] # (T, num_types)
        excite = a * jnp.exp(-alpha * dist[:, None])
        excite = jnp.where(mask[:, None], excite, 0.) # (T, num_types)
        excite = excite.sum(0) # (num_types, )
        intensities = mu + excite
        return intensities # (num_types, )
    
    def _draw_next(self, ts: Float[Array, "T"], marks: Int[Array, "T"], mask: Bool[Array, "T"], dt_max: float, num_samples: int, over_sample_rate: float, key):
        bound_sample_dts = jnp.linspace(0., dt_max, num_samples) # (num_samples, )
        bound_sample_intensities = jax.vmap(self.intensities_at, (None, None, None, 0))(ts, marks, mask, bound_sample_dts) # (num_samples, num_types)
        bound = bound_sample_intensities.max(0) # (num_types, )
        sample_rate = bound * over_sample_rate # (num_types, )

        next_dt_samples = list()
        num_types = self.params[0].shape[0]
        for i in range(num_types):
            t = 0
            while True:
                cur_key, key = jax.random.split(key)
                exp_numbers = jax.random.exponential(cur_key, shape=(num_samples, )) # (num_samples, )
                exp_numbers = exp_numbers / sample_rate[i] # (num_samples, )
                sample_dt = exp_numbers.cumsum(-1) + t # (num_samples, )
                sample_intensities = jax.vmap(self.intensities_at, (None, None, None, 0))(ts, marks, mask, sample_dt) # (num_samples, num_types)
                sample_intensity = sample_intensities[:, i] # (num_samples, )
                cur_key, key = jax.random.split(key)
                accept_ind = self._sample_accept(sample_intensity, sample_rate[i], cur_key)
                if accept_ind>=0:
                    next_dt_samples.append(sample_dt[accept_ind])
                    break
                t = sample_dt[-1]
        next_dt_samples = jnp.asarray(next_dt_samples)
        next_mark = next_dt_samples.argmin(-1)
        next_dt = next_dt_samples[next_mark]
        return next_dt, next_mark

    @eqx.filter_jit
    def _sample_accept(self, sample_intensity: Float[Array, "num_samples"],  sample_rate: float, key):
        num_samples = sample_intensity.shape[0]
        unif_numbers = jax.random.uniform(key, (num_samples, ))
        criterion = unif_numbers * sample_rate / sample_intensity # (num_samples, )
        accept = criterion<=1
        ind = accept.argmax(-1) # ()
        has_accept = jnp.any(accept)
        ind = jnp.where(has_accept, ind, -1)
        return ind

    def sample(self, t_max: float, key, **kwargs): # CPU function
        _, _, mu = self.params # (num_types, )
        cur_key, key = jax.random.split(key)
        num_types = len(mu)
        exp_numbers = jax.random.exponential(cur_key, (num_types, )) / mu
        mark = exp_numbers.argmin(-1)
        t = exp_numbers[mark]
        ts = [t.item(), ]
        marks = [mark.item(), ]
        assert t<t_max
        while True:
            ts_array = jnp.asarray(ts)
            marks_array = jnp.asarray(marks)
            cur_key, key = jax.random.split(key)
            dt, mark = self._draw_next(ts_array, marks_array, jnp.ones_like(marks_array, dtype=bool), kwargs['dt_max'], kwargs['num_samples'], kwargs['over_sample_rate'], cur_key)
            t = t + dt
            if t>=t_max:
                break
            ts.append(t.item())
            marks.append(mark.item())
        ts = np.asarray(ts)
        marks = np.asarray(marks)
        return ts, marks
    
if __name__ == '__main__':
    mus = jnp.asarray([.5, .4, .3])
    As = jnp.asarray([[1., .5, .5], [.5, 1, .5], [.5, .5, 1.]])
    aa = jnp.asarray([[1., 2., 2.], [2., 1., 2.], [2., 2., 1.]])*2
    # poisson = Poisson(mus)
    hawkes = Hawkes((As, aa, mus))
    key = jax.random.PRNGKey(0)
    # seq = poisson.sample(100., key)
    seq = hawkes.sample(10., key, dt_max=10., num_samples=2000, over_sample_rate=5.)
    print()