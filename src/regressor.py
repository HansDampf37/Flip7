from torch import nn

from src.flip7_game import Flip7Game


class Flip7Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(Flip7Game.NUM_CARDS, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(p=0.1),

            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(p=0.1),

            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(p=0.1),

            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(p=0.3),

            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.model.forward(x)