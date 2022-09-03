from torch.utils.data import IterableDataset
import numpy as np
import random

class MultVideoItrable(IterableDataset):
    """
    Iterable dataset that combines multiple videos and is compatible with pytorch
    """
    def __init__(self, videos: list):
        super().__init__()
        self.videos = list()
        for v in videos:
            self.videos.append(iter(v))
        self.options = list(range(len(self.videos)))
        
    def get_content(self):
        while len(self.options)>0:
            rv = random.choice(self.options)
            try:
                vid = next(self.videos[rv])
                yield self.proc_seq(vid)
            except:
                self.options.remove(rv)

    def proc_seq(self, seq):
        return seq
        
    def __iter__(self):
        return iter(self.get_content())

class ProcMultVideoItrable(MultVideoItrable):
    """
    Dataset that also normalizes data
    """
    def __init__(self, mean, std, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean = self.match_dims(mean)
        self.std = self.match_dims(std)
        self.type = np.float32

    def proc_seq(self, seq):
        rese = np.transpose(seq[0], (0,3,1,2))
        rese = (rese-self.mean)/self.std
        return rese.astype(self.type), seq[1].astype(self.type)

    @staticmethod
    def match_dims(arr):
        return np.expand_dims(arr, axis=[0,2,3])
