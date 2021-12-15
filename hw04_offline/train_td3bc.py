import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import tqdm
import copy


ALPHA = 2.5
GAMMA = 0.99
TAU = 0.002
CRITIC_LR = 5e-4
ACTOR_LR = 2e-4
BATCH_SIZE = 256
TRANSITIONS = 1_000_000
RANDOM_SEED = 12
NOISE_EPS = 0.2
NORM_EPS = 1e-3


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


def soft_update(target, source):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - TAU) * tp.data + TAU * sp.data)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.model(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=-1)).view(-1)


class TD3BC:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic_1 = Critic(state_dim, action_dim)
        self.critic_2 = Critic(state_dim, action_dim)

        self.actor_optim = Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_1_optim = Adam(self.critic_1.parameters(), lr=CRITIC_LR)
        self.critic_2_optim = Adam(self.critic_2.parameters(), lr=CRITIC_LR)

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)

    def update(self, batch):
        # Sample batch
        state, action, next_state, reward, done = batch
        state = torch.tensor(np.array(state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        done = torch.tensor(np.array(done), dtype=torch.float)

        # Update critic
        with torch.no_grad():
            next_action = self.target_actor(next_state)
            noise = NOISE_EPS * torch.randn_like(action)
            next_action = torch.clamp(next_action + noise, -1, +1)

            # Compute Q target values
            target_q_1 = self.target_critic_1(next_state, next_action)
            target_q_2 = self.target_critic_2(next_state, next_action)
            q_target = torch.min(target_q_1, target_q_2)
            q_target = reward + (1 - done) * GAMMA * q_target

        # Update first critic
        current_q_1 = self.critic_1(state, action)
        critic_loss_1 = F.mse_loss(current_q_1, q_target)
        self.critic_1_optim.zero_grad()
        critic_loss_1.backward()
        self.critic_1_optim.step()

        # Update second critic
        current_q_2 = self.critic_2(state, action)
        critic_loss_2 = F.mse_loss(current_q_2, q_target)
        self.critic_2_optim.zero_grad()
        critic_loss_2.backward()
        self.critic_2_optim.step()

        # Update actor
        policy = self.actor(state)
        Q_values = self.critic_1(state, policy)
        lmbda = ALPHA / Q_values.abs().mean().detach()

        actor_loss = -lmbda * Q_values.mean() + F.mse_loss(policy, action)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Update target models
        soft_update(self.target_critic_1, self.critic_1)
        soft_update(self.target_critic_2, self.critic_2)
        soft_update(self.target_actor, self.actor)

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state]), dtype=torch.float)
            return self.actor(state).numpy()[0]

    def save(self):
        torch.save(self.actor, "agent.pkl")


def train():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    net = TD3BC(19, 5)
    dataset = TransitionDataset("optimal.npz")
    states, next_states = [], []
    for i in range(len(dataset)):
        states.append(dataset.transitions[i][0])
        next_states.append(dataset.transitions[i][2])

    states = np.array(states)
    next_states = np.array(next_states)
    mean = states.mean(axis=0)
    std = states.std(axis=0) + NORM_EPS
    normalized_states = (states - mean) / std
    normalized_next_states = (next_states - mean) / std

    for i in range(len(dataset)):
        dataset.transitions[i][0] = normalized_states[i]
        dataset.transitions[i][2] = normalized_next_states[i]

    for i in tqdm.tqdm(range(TRANSITIONS)):
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        for batch in dataloader:
            net.update(batch)

        if i % 10000 == 0:
            net.save()

    return net


if __name__ == "__main__":
    model = train()
