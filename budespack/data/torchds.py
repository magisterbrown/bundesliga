from torch.utils.data import IterableDataset
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
                yield vid
            except:
                self.options.remove(rv)
        
    def __iter__(self):
        return iter(self.get_content())
