from jax_dataloader import Dataset, manual_seed, DataLoader
import numpy as np
import pickle

class TPPSeqData(Dataset):
    def __init__(self, data, set_max_len = -1, max_seqs = -1, num_type=None):
        max_len = max([len(seq) for seq in data['marks']])
        self.max_len = max_len
        self.marks = np.asarray(pad2(data['marks'], max_len), dtype=int)
        self.times = np.asarray(pad2(data['times'], max_len), dtype=float)
        self.mask = np.asarray([[1]*len(seq)+[0]*(max_len-len(seq)) for seq in data['marks']], dtype=bool)
        if set_max_len > 0:
            self.marks = self.marks[:, :set_max_len]
            self.times = self.times[:, :set_max_len]
            self.mask = self.mask[:, :set_max_len]
            self.max_len = max_len
        if max_seqs > 0:
            self.marks = self.marks[:max_seqs, :]
            self.times = self.times[:max_seqs, :]
            self.mask = self.mask[:max_seqs, :]
        if num_type is not None and num_type >0:
            max_types = self.marks.max(-1) # (num_seqs, )
            valid_inds = np.where(max_types<num_type)
            self.marks = self.marks[valid_inds]
            self.times = self.times[valid_inds]
            self.mask = self.mask[valid_inds]

    def __len__(self):
        return self.marks.shape[0]
    
    def __getitem__(self, index):
        return self.times[index], self.marks[index], self.mask[index]
    
    def dt_stat(self):
        dts = self.times[:, 1:] - self.times[:, :-1]
        dts = dts[self.mask[:, 1:]]
        dt_max = dts.max()
        dt_mean = dts.mean()
        dt_std = dts.std()
        dt_min = dts.min()
        return dt_max.item(), dt_mean.item(), dt_std.item(), dt_min.item()
    
    def all_dts(self):
        dts = self.times[:, 1:] - self.times[:, :-1]
        dts = dts[self.mask[:, 1:]].tolist()
        return dts
    
    def get_dtmax(self):
        dts = self.times[:, 1:] - self.times[:, :-1]
        dts = dts[self.mask[:, 1:]]
        dt_max = dts.max()
        return dt_max.item()

    def get_dataloader(self, batch_size, shuffle=False):
        dataloader = DataLoader(self, 'jax', batch_size, shuffle)
        return dataloader
    
    def num_types(self):
        return self.marks.max().item()+1
    
    def max_length(self):
        return self.max_len
    
class TPPSliceData(Dataset):
    def __init__(self, data):
        self.hist_len = len(data['hist_marks'][0])
        self.pred_len = len(data['pred_marks'][0])
        self.hist_ts = np.asarray(data['hist_times'])
        self.hist_ms = np.asarray(data['hist_marks'])
        self.pred_ts = np.asarray(data['pred_times'])
        self.pred_ms = np.asarray(data['pred_marks'])

    def __len__(self):
        return self.hist_ts.shape[0]

    def __getitem__(self, index):
        return (self.hist_ts[index], self.hist_ms[index]), (self.pred_ts[index], self.pred_ms[index])
    
    def get_dataloader(self, batch_size, shuffle=False):
        dataloader = DataLoader(self, 'jax', batch_size, shuffle)
        return dataloader
    
    def num_types(self):
        return max([self.hist_ms.max()+1, self.pred_ms.max()+1]).item()

    def dt_stat(self):
        ts = np.concatenate((self.hist_ts, self.pred_ts), 1)
        dts = ts[:, 1:] - ts[:, :-1]
        dt_max = dts.max()
        dt_mean = dts.mean()
        dt_std = dts.std()
        return dt_max.item(), dt_mean.item(), dt_std.item()

def shuffle_seed(seed):
    manual_seed(seed)

def pad2(seqs, max_len):
    out_seqs = np.array([
        seq + [0] * (max_len - len(seq))
        for seq in seqs
    ])
    return out_seqs

def load_data_hym(name, split, max_len=-1, max_seqs=-1, num_types=None):
    with open(f'my_datasets/{name}/{split}.pkl', 'rb') as f:
        data = pickle.load(f, encoding='latin-1')
    data = data[split]
    time = [[elem['time_since_start']-inst[0]['time_since_start'] for elem in inst] for inst in data]
    mark = [[elem['type_event'] for elem in inst] for inst in data]
    dataset = TPPSeqData({'times': time, 'marks': mark}, max_len, max_seqs, num_types)
    return dataset

