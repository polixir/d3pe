import torch
import numpy as np

import neorl

from typing import *

class OPEDataset(torch.utils.data.Dataset):
    def __init__(self, data : Dict[str, np.ndarray], start_indexes : Optional[np.ndarray] = None) -> None:
        super().__init__()
        self.data = data
        self.total_size = self.data['obs'].shape[0]
        self.start_indexes = start_indexes
        if self.has_trajectory:
            self.trajectory_number = self.start_indexes.shape[0]
            self.end_indexes = np.concatenate([self.start_indexes[1:], np.array([self.total_size])])
    
    @property
    def has_trajectory(self) -> bool:
        return self.start_indexes is not None

    def get_trajectory(self,) -> List[Dict[str, np.ndarray]]:
        assert self.has_trajectory
        return [{k : v[start : end] for k, v in self.data.items()} for start, end in zip(self.start_indexes, self.end_indexes)]

    def get_initial_states(self,) -> Dict[str, np.ndarray]:
        assert self.has_trajectory
        return {k : v[self.start_indexes] for k, v in self.data.items()}

    def sample(self, batch_size) -> Dict[str, np.ndarray]:
        indexes = np.random.randint(0, self.total_size, size=(batch_size))
        return {k : v[indexes] for k, v in self.data.items()}

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, index) -> Dict[str, np.ndarray]:
        return {k : v[index] for k, v in self.data.items()}

def get_neorl_datasets(task : str, level : str, amount : int) -> Tuple[OPEDataset, OPEDataset]:
    env = neorl.make(task)
    train_data, val_data = env.get_dataset(data_type=level, train_num=amount)
    train_start_indexes = train_data.pop('index')
    val_start_indexes = val_data.pop('index')
    return (OPEDataset(train_data, train_start_indexes), OPEDataset(val_data, val_start_indexes))

def to_torch(data : dict, dtype = torch.float32, device = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict[str, torch.Tensor]:
    return {k : torch.as_tensor(v, dtype=dtype, device=device) for k, v in data.items()}

def to_numpy(data : dict) -> Dict[str, np.ndarray]:
    return {k : v if isinstance(v, np.ndarray) else v.detach().cpu().numpy() for k, v in data.items()}