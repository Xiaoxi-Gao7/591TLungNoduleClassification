import torch.nn as nn
from torchvision import models

def get_model(name="densenet121"):
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    num_ftrs = model.classifier.in_features
    # regularization
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5), 
        nn.Linear(num_ftrs, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 2)
    )
    return model