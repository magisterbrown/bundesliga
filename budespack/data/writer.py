import os
import tarfile
import re
from budespack.data.original import Timer, VideoReader
from budespack.data.iterative import VidBatcher
import numpy as np
import pandas as pd

class PulWork:
    def __init__(self, df: pd.DataFrame, matches: dict, dadir: str):
        self.df = df
        self.matches = matches
        self.dadir = dadir
        os.makedirs(dadir, exist_ok=True)
        
    def reocrd_entry(self, path):
        matches = re.findall('\/([\w\d_]*)\.mp4',path)
        assert len(matches) == 1
        ide = matches[0]
        esxist = self.df[self.df['video_id']==ide]
        
        timer = Timer(esxist,1)
        vrd = VideoReader(path, timer, 384, self.matches)
        self.write_video(vrd, ide)
        
        return ide
    
    def write_video(self, vrd,idx):
        bs = 16
        vli = VidBatcher(bs,vrd)
        totdir = f'{self.dadir}/{idx}'
        os.makedirs(totdir, exist_ok=True)

        for key,el in enumerate(vli):
            np.savez(f'{totdir}/{key}_{idx}.npz', vid=el[0].astype(np.uint8), lab=el[1].astype(np.uint8))
            break
        
        with tarfile.open(f"{self.dadir}/{idx}.tar", "w") as tar:
            for name in os.listdir(f'{totdir}/'):
                tar.add(f'{totdir}/{name}',arcname=name)
                
        print(f'{idx} done')
