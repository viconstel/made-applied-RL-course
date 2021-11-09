import random
import numpy as np
import os
import torch


class Agent:
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = torch.load(__file__[:-8] + "/agent.pkl").to(self.device)
        
    def act(self, state):
        state = torch.tensor(np.array(state, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            Q_values = self.model(state)
        return Q_values.argmax().cpu().numpy()

