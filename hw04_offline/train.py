import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


RANDOM_SEED = 12
LR = 1e-3
BATCH_SIZE = 256
MODEL_FILE_PATH = 'agent.pkl'
SUBOPTIMAL_STEPS = 500
OPTIMAL_STEPS = 1200


class TransitionDataset(Dataset):
    def __getitem__(self, index):
        state, action, next_state, reward, done = self.transitions[index]
        return torch.tensor(state, dtype=torch.float32), torch.tensor(action, dtype=torch.float32), \
               torch.tensor(next_state, dtype=torch.float32), torch.tensor([reward], dtype=torch.float32), \
               torch.tensor([done], dtype=torch.float32)

    def __init__(self, path):
        self.transitions = np.load(path, allow_pickle=True)["arr_0"]

    def __len__(self):
        return len(self.transitions)


class BehavioralCloning(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )

    def act(self, state):
        return self.model(state)

    def save(self):
        torch.save(self.model, MODEL_FILE_PATH)


def train():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    net = BehavioralCloning(input_size=19, output_size=5, hidden_size=256)
    optim = torch.optim.Adam(net.parameters(), lr=LR)

    dataset = TransitionDataset('suboptimal.npz')
    for i in range(SUBOPTIMAL_STEPS):
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        for batch in dataloader:
            state, action, _, _, _ = batch
            action_pred = net.act(state)
            loss = F.mse_loss(action_pred, action)
            optim.zero_grad()
            loss.backward()
            optim.step()
        print(f'Iter:{i}/{SUBOPTIMAL_STEPS}, suboptimal loss value: {loss}')

    dataset = TransitionDataset('optimal.npz')
    for i in range(OPTIMAL_STEPS):
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        for batch in dataloader:
            state, action, _, _, _ = batch
            action_pred = net.act(state)
            loss = F.mse_loss(action_pred, action)
            optim.zero_grad()
            loss.backward()
            optim.step()
        print(f'Iter:{i}/{OPTIMAL_STEPS}, optimal loss value: {loss}')

    net.save()
    return net


if __name__ == '__main__':
    model = train()
