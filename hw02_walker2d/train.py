import pybullet_envs
# Don't forget to install PyBullet!
from gym import make
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F
from torch.optim import Adam
import random

ENV_NAME = "Walker2DBulletEnv-v0"

LAMBDA = 0.95
GAMMA = 0.99

ACTOR_LR = 2e-4
CRITIC_LR = 1e-4

CLIP = 0.2
ENTROPY_COEF = 1e-2
BATCHES_PER_UPDATE = 128
BATCH_SIZE = 512
HIDDEN_SIZE = 256

MIN_TRANSITIONS_PER_UPDATE = 4096
MIN_EPISODES_PER_UPDATE = 10

ITERATIONS = 1000
RANDOM_SEED = 12

    
def compute_lambda_returns_and_gae(trajectory):
    lambda_returns = []
    gae = []
    last_lr = 0.
    last_v = 0.
    for _, _, r, _, v in reversed(trajectory):
        ret = r + GAMMA * (last_v * (1 - LAMBDA) + last_lr * LAMBDA)
        last_lr = ret
        last_v = v
        lambda_returns.append(last_lr)
        gae.append(last_lr - v)
    
    # Each transition contains state, action, old action probability,
    # value estimation and advantage estimation
    zip_object = zip(trajectory, reversed(lambda_returns), reversed(gae))
    return [(s, a, p, v, adv) for (s, a, _, p, _), v, adv in zip_object]


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Advice: use same log_sigma for all states to improve stability
        # Do this by defining log_sigma as nn.Parameter(torch.zeros(...))
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = nn.Sequential(
            nn.Linear(self.state_dim, 2 * HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(2 * HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, self.action_dim)
        )
        self.sigma = nn.Parameter(torch.zeros(self.action_dim))
        
    def compute_proba(self, state, action):
        # Returns probability of action according
        # to current policy and distribution of actions
        mu = self.model(state)
        sigma = torch.exp(self.sigma)
        distribution = Normal(mu, sigma)
        return torch.exp(distribution.log_prob(action).sum(-1)), distribution
        
    def act(self, state):
        # Returns an action (with tanh), not-transformed action (without tanh)
        # and distribution of non-transformed actions
        # Remember: agent is not deterministic,
        # sample actions from distribution (e.g. Gaussian)
        mu = self.model(state)
        sigma = torch.exp(self.sigma)
        distribution = Normal(mu, sigma)
        action = distribution.sample()

        return torch.tanh(action), action, distribution
        
        
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )
        
    def get_value(self, state):
        return self.model(state)


class PPO:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optim = Adam(self.actor.parameters(), ACTOR_LR)
        self.critic_optim = Adam(self.critic.parameters(), CRITIC_LR)

    def update(self, trajectories):
        # Turn a list of trajectories into list of transitions
        transitions = [t for traj in trajectories for t in traj]
        state, action, old_prob, target_value, advantage = zip(*transitions)
        state = np.array(state)
        action = np.array(action)
        old_prob = np.array(old_prob)
        target_value = np.array(target_value)
        advantage = np.array(advantage)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        for _ in range(BATCHES_PER_UPDATE):
            # Choose random batch
            idx = np.random.randint(0, len(transitions), BATCH_SIZE)
            s = torch.tensor(state[idx]).float()
            a = torch.tensor(action[idx]).float()
            # Probability of the action in state s.t. old policy
            op = torch.tensor(old_prob[idx]).float()
            # Estimated by lambda-returns
            v = torch.tensor(target_value[idx]).float()
            # Estimated by generalized advantage estimation
            adv = torch.tensor(advantage[idx]).float()

            prob, dist = self.actor.compute_proba(s, a)
            importance = prob / op
            clipped_importance = torch.clamp(importance, 1 - CLIP, 1 + CLIP)
            actor_loss = -1. * (torch.min(importance * adv, clipped_importance * adv).mean())
            critic_loss = F.mse_loss(self.critic.get_value(s).flatten(), v)
            actor_loss = actor_loss - ENTROPY_COEF * dist.entropy().mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

    def get_value(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state])).float()
            value = self.critic.get_value(state)
        return value.cpu().item()

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state])).float()
            action, pure_action, distr = self.actor.act(state)
            prob = torch.exp(distr.log_prob(pure_action).sum(-1))
        return action.cpu().numpy()[0], pure_action.cpu().numpy()[0], prob.cpu().item()

    def save(self):
        torch.save(self.actor, "agent.pkl")


def evaluate_policy(env, agent, episodes=5):
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        
        while not done:
            state, reward, done, _ = env.step(agent.act(state)[0])
            total_reward += reward
        returns.append(total_reward)
    return returns
   

def sample_episode(env, agent):
    s = env.reset()
    d = False
    trajectory = []
    while not d:
        a, pa, p = agent.act(s)
        v = agent.get_value(s)
        ns, r, d, _ = env.step(a)
        trajectory.append((s, pa, r, p, v))
        s = ns
    return compute_lambda_returns_and_gae(trajectory)


if __name__ == "__main__":
    env = make(ENV_NAME)
    env.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    ppo = PPO(state_dim=env.observation_space.shape[0],
              action_dim=env.action_space.shape[0])
    state = env.reset()
    episodes_sampled = 0
    steps_sampled = 0
    best_score = 0.
    
    for i in range(ITERATIONS):
        trajectories = []
        steps_ctn = 0
        
        while len(trajectories) < MIN_EPISODES_PER_UPDATE or steps_ctn < MIN_TRANSITIONS_PER_UPDATE:
            traj = sample_episode(env, ppo)
            steps_ctn += len(traj)
            trajectories.append(traj)
        episodes_sampled += len(trajectories)
        steps_sampled += steps_ctn

        ppo.update(trajectories)        
        
        if (i + 1) % (ITERATIONS//100) == 0:
            rewards = evaluate_policy(env, ppo, 5)
            print(f"Step: {i+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}, Episodes: {episodes_sampled}, Steps: {steps_sampled}")
            ppo.save()
            if np.mean(rewards) > best_score:
                best_score = np.mean(rewards)
                torch.save(ppo.actor, "best.pkl")
