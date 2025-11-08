import equinox as eqx
from jaxtyping import PyTree, Float, Array, Int, Bool
from models.itpp1 import ITPP as ITPP1



class MTPP(eqx.Module):
    model: eqx.Module
    model_name: str

    def __init__(self, model_name: str, num_types: int, 
                                hdim: int, key: Array, args: dict):
        if model_name == 'itpp':
            self.model = ITPP1(hdim, num_types, args['nhead'], args['reg'], key)
        else:
            raise NotImplementedError()
 
        self.model_name = model_name

    @eqx.filter_jit
    def __call__(self, ts: Float[Array, "T"], marks: Int[Array, "T"], mask: Bool[Array, "T"], key: Array=None):
        out = self.model(ts, marks, mask, key)
        return out
    
    @eqx.filter_jit
    def sequence_one_step_predict(self, ts: Float[Array, "T"], marks: Int[Array, "T"], mask: Bool[Array, "T"], key: Array, dt_max: float):
        predict_tuple, real_tuple, mask = self.model.rolling_predict(ts, marks, mask, dt_max, key)
        return predict_tuple, real_tuple, mask
    
    @eqx.filter_jit
    def encode(self, ts: Float[Array, "T"], marks: Int[Array, "T"], mask: Bool[Array, "T"], key=None):
        states = self.model.encode(ts, marks, mask, key)
        return states
    
    @eqx.filter_jit
    def intensities_at(self, state: PyTree, dts: Float[Array, "num_samples"], key=None):
        intensities = self.model.intensities_at(state, dts, key)
        return intensities
