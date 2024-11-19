import torch
from torch import nn
from models.S_Former import spatial_transformer
#from S_Former import spatial_transformer
from models.T_Former import temporal_transformer
#from T_Former import temporal_transformer


class GenerateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.s_former = spatial_transformer()
        self.t_former = temporal_transformer()
        self.fc = nn.Linear(512, 7)
        self.AU_fc = nn.Linear(512, 18)

    def forward(self, x):

        x = self.s_former(x)
        x = self.t_former(x)
        AU_pred = self.AU_fc(x)
        x = self.fc(x)
        return x,AU_pred


if __name__ == '__main__':
    img = torch.randn((1, 16, 3, 112, 112))
    model = GenerateModel()
    model(img)
