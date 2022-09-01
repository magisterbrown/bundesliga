import pandas as pd
import numpy as np
import cv2

class Timer:
    """
    Timer contains order of seconds that should be loaded.
    """
    def __init__(self, time: pd.DataFrame, max_step: float):

        timer = time.iterrows()
        curr, event = self.get_entry(next(timer))
        self.items = [(curr, event)]
        for item in timer:
            item, event = self.get_entry(item)
            while item-curr>max_step:
                curr+=max_step
                self.items.append((curr,False))

            self.items.append((item,event))


    def get_entry(self, nx):
        nx = nx[1]
        return nx['time'], nx['event']

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def __len__(self):
        return len(self.items)

class VideoReader:
    """
    Reads frames from video that exist in timer.
    """
    def __init__(self, video: str, times: Timer, size: int, maper: dict):
        self.video = cv2.VideoCapture(video)
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.times = times
        self.size = size
        self.maper = maper
        
    def proc_frame(self, frame):
        frame = cv2.resize(frame, (self.size, self.size), interpolation = cv2.INTER_AREA)
        return frame
    
    def proc_lable(self, event):
        lable = np.zeros(len(self.maper))
        if event:
            lable[self.maper[event]]=1
        return lable
    
    def __getitem__(self, idx):
        time, event = self.times[idx]
        self.video.set(cv2.CAP_PROP_POS_FRAMES, int(self.fps*time))

        ret, frame = self.video.retrieve()
        return self.proc_frame(frame), self.proc_lable(event)
    
    def __len__(self):
        return len(self.timer)
    
    def __del__(self):
        self.video.release()
