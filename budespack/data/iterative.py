import numpy as np
from budespack.data.original import VideoReader

class VidoeLocalIterable:
    """
    Iterate over video reder to create iterator
    """
    def __init__(self, vid: VideoReader):
        self.vid = vid
        
    def get_content(self):
        for el in self.vid:
            yield el
    
    def __iter__(self):
        return iter(self.get_content())

class VidBatcher(VidoeLocalIterable):
    """
    Iterates over video and packs it into the batches
    """
    def __init__(self, bs: int ,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bs = bs
        
    def get_content(self):
        cnt = 0
        npa = list()
        lables = list()
        for el in iter(super().get_content()):
            npa.append(el[0])
            lables.append(el[1])
            cnt+=1
            if cnt==self.bs:
                yield np.stack(npa), np.stack(lables)
                cnt=0
                npa = list()
                lables = list()
            
