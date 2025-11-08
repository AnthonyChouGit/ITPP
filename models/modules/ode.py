from diffrax import diffeqsolve, Dopri5, ODETerm, PIDController, BacksolveAdjoint, Tsit5, ConstantStepSize, VirtualBrownianTree, MultiTerm, ControlTerm, Euler, ReversibleHeun, SaveAt
from jaxtyping import Float, Array, PyTree
import jax
import equinox as eqx
import jax.numpy as jnp

@eqx.filter_jit
def integrate(func: callable, t0: Float[Array, "1"], t1: Float[Array, "1"], x0: PyTree, args: PyTree):
    solution = diffeqsolve(
        ODETerm(func), 
        # Dopri5(),
        Tsit5(),
        t0,
        t1,
        None,
        x0,
        args,
        stepsize_controller=PIDController(rtol=1e-3, atol=1e-6),
        adjoint=BacksolveAdjoint(),
        max_steps=2 ** 31 - 1,
    )
    # x1 = solution.ys[-1]
    x1 = jax.tree.map(lambda x: x[-1], solution.ys)
    return x1

@eqx.filter_jit
def integrate_save(func: callable, t0: Float[Array, "1"], save_at: Float[Array, "num_samples"], x0: PyTree, args: PyTree):
    t1 = save_at[-1]
    solution = diffeqsolve(
        ODETerm(func), 
        Tsit5(),
        t0,
        t1,
        None,
        x0,
        args,
        stepsize_controller=PIDController(rtol=1e-3, atol=1e-6),
        adjoint=BacksolveAdjoint(),
        max_steps=2 ** 31 - 1,
        saveat=SaveAt(ts=save_at)
    )
    x1 = solution.ys # (num_samples, ...)
    return x1

@eqx.filter_jit
def integrate2(func: callable, t0: Float[Array, "1"], t1: Float[Array, "1"], x0: PyTree, args: PyTree):
    solution = diffeqsolve(
        ODETerm(func), 
        Dopri5(),
        # Tsit5(),
        t0,
        t1,
        .0005*(t1-t0),
        x0,
        args,
        stepsize_controller=ConstantStepSize(),
        adjoint=BacksolveAdjoint(),
        max_steps=2 ** 31 - 1,
    )
    # x1 = solution.ys[-1]
    x1 = jax.tree.map(lambda x: x[-1], solution.ys)
    return x1

class MultiBM(VirtualBrownianTree):
    num: int

    def __init__(self, t0, t1, tol, dim, aux_num: int, key):
        super().__init__(t0, t1, tol, shape=(dim, ), key=key)
        self.num = aux_num

    
    def evaluate(
            self,
            t0,
            t1 = None,
            left: bool = True,
            use_levy: bool = False,
        ):
        bm = super().evaluate(t0, t1, left, use_levy) # (dim, )
        out = [jnp.asarray(0.)] * self.num + [bm, ]
        return tuple(out)

@eqx.filter_jit
def integrate_sde(drift: callable, control: callable, t0: Float[Array, ""], t1: Float[Array, ""], x0: PyTree, args: PyTree, num_steps: int, key):
    dt = (t1-t0) / num_steps
    bmt = MultiBM(t0, t1+1e-4, 0.5*dt, x0[-1].shape[-1], 1, key)
    # bm = VirtualBrownianTree(t0, t1, .0025 * (t1 - t0), shape=(), key=key)
    term = MultiTerm(ODETerm(drift), ControlTerm(control, bmt))
    solution = diffeqsolve(
        term, 
        # Euler(), 
        ReversibleHeun(),
        t0, 
        t1, 
        dt,
        x0,
        args,
        adjoint=BacksolveAdjoint(),
        max_steps=2 ** 31 - 1
    )
    x1 = jax.tree.map(lambda x: x[-1], solution.ys)
    return x1
