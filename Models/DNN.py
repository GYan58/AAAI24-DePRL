import torch.nn.functional as F
from torchvision import models

from Settings import *

        
class dnn_har(nn.Module):
    def __init__(self):
        super(dnn_har, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(900, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.05),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 5),
        )

        self.to(device)
        
    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.features(out)
        out = self.classifier(out)
        return out
   
        
