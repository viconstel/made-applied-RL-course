from gym import make
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from collections import deque
import random
import copy

GAMMA = 0.99
INITIAL_STEPS = 2048
TRANSITIONS = 750000
STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
RANDOM_SEED = 12
HIDDEN_SIZE = 256


class DQN:
    def __init__(self, state_dim, action_dim, device):
        self.steps = 0  # Do not change
        self.buffer = deque(maxlen=TRANSITIONS)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.model = nn.Sequential(
            nn.Linear(self.state_dim, 2 * HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(2 * HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, self.action_dim)
        ).to(self.device)

        self.target_model = copy.deepcopy(self.model)
        self.optimizer = Adam(self.model.parameters(), LEARNING_RATE)

    def consume_transition(self, transition):
        self.buffer.append(transition)

    def sample_batch(self):
        batch = random.sample(self.buffer, BATCH_SIZE)
        return list(zip(*batch))

    def train_step(self, batch):
        state, action, next_state, reward, done = batch
        state = torch.tensor(np.array(state, dtype=np.float32)).to(self.device)
        action = torch.tensor(np.array(action, dtype=np.int64)).to(self.device)
        next_state = torch.tensor(np.array(next_state, dtype=np.float32)).to(
            self.device)
        reward = torch.tensor(np.array(reward, dtype=np.float32)).to(self.device)
        done = torch.tensor(np.array(done, dtype=np.bool8))

        Q_values = self.model(state).gather(1, action.reshape(-1, 1)).cpu().squeeze()

        with torch.no_grad():
            Q_prime_values = self.target_model(next_state).cpu()

        target_values = torch.max(Q_prime_values, dim=1).values
        target_values[done] = 0.
        target_values = reward + GAMMA * target_values

        loss = F.mse_loss(Q_values, target_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state, target=False):
        state = torch.tensor(np.array(state, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            Q_values = self.model(state)
        return Q_values.argmax().cpu().numpy()

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    def save(self):
        torch.save(self.model, "agent.pkl")


def evaluate_policy(agent, episodes=5):
    env = make("LunarLander-v2")
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.

        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return returns


if __name__ == "__main__":
    env = make("LunarLander-v2")
    env.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dqn = DQN(state_dim=env.observation_space.shape[0],
              action_dim=env.action_space.n, device=device)
    eps = 0.1
    state = env.reset()

    for _ in range(INITIAL_STEPS):
        action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        dqn.consume_transition((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

    for i in range(TRANSITIONS):
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        next_state, reward, done, _ = env.step(action)
        dqn.update((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

        if (i + 1) % (TRANSITIONS // 100) == 0:
            rewards = evaluate_policy(dqn, 5)
            print(f"Step: {i + 1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            dqn.save()
