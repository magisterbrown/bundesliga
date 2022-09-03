import torch.nn as nn
from torchvision.models import efficientnet_v2_s
import torch

class Final(nn.Module):
    """
    Applies classifier layer to every gru output
    """
    def __init__(self, inp, hid, out):
        super().__init__()
        self.l1 = nn.Linear(inp,hid)
        self.prl = nn.PReLU()
        self.l2 = nn.Linear(hid,out)
        
    def layer_sing(self, x):
        x = self.l1(x)
        x = self.prl(x)
        x = self.l2(x)
        
        return x
    
    def forward(self, x):
        sh = x.shape
        outs = sh[0]*sh[1]
        parel = sh[1]
        result = torch.empty((outs,5),device=x.device)
        for i,entry in enumerate(x):
            sepr = self.layer_sing(entry)
            result[i*parel:i*parel+parel] = sepr

        return result

class VideoClassifier(nn.Module):
    """
    Classifies sequence of the frames.
    """
    def __init__(self, rec: int, side: int, hidd_gru: int, hid_fin: int, layers_gru=1):
        super().__init__()
        effnet = efficientnet_v2_s()
        self.features = effnet.features
        self.pool = effnet.avgpool
        self.rec = rec
        self.side = side
        
        self.hidd_gru = hidd_gru
        self.gru = nn.GRU(1280,hidd_gru,layers_gru,batch_first=True)
        self.classifier = Final(hidd_gru,hid_fin,5)
    
    def forward(self, x):
        x = x.view(-1,3,self.side,self.side)
        x = self.features(x)
        x = self.pool(x)
        x = x.view(-1,self.rec,1280)
        x = self.gru(x)[0]
        x = self.classifier(x)
        return x
    
