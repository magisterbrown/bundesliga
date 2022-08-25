import pandas as pd
import numpy as np
import cv2

class Timer:
    def __init__(self, time: pd.Series, max_step: float):
        self.time = time
        self.max_step = max_step 
        
    def __iter__(self):
        self.pos = 0
        self.value = self.time.iloc[self.pos]
        return self
    
    def __next__(self):
        
        if(self.pos>=len(self.time)):
            raise StopIteration
        retv = self.value
        
        if(self.pos+1>=len(self.time)):
            self.pos+=1
            return retv
        
        if self.time.iloc[self.pos+1]-retv > self.max_step:
            self.value+=self.max_step
        else:
            self.pos+=1
            self.value=self.time.iloc[self.pos]
        
        return retv    

class VideoReader:
    def __init__(self, video: cv2.VideoCapture, times, size: int):
        self.video = cv2.VideoCapture(video)
        self.timer = times
        self.curr_frame = 0
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.size = size

    def proc_frame(self, frame):
        frame = cv2.resize(frame, (self.size, self.size), interpolation = cv2.INTER_AREA)
        return frame

    def __next__(self):
        time = next(self.timer)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, int(self.fps*time))

        ret, frame = self.video.retrieve()
        return self.proc_frame(frame)

    def __iter__(self):
        return self

    def __del__(self):
        self.video.release()
