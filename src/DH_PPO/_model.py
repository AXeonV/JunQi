import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()):
	device = torch.device('cuda:0')
	torch.cuda.empty_cache()
	print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
	print("Device set to : cpu")
print("============================================================================================")

class RolloutBuffer:
	def __init__(self):
		self.states = []
		self.actions_id = []
		self.actions_to = []
		self.logprobs_id = []
		self.logprobs_to = []
		self.rewards = []
		self.state_values = []
		self.is_terminals = []
		self.valid_id = []
		self.valid_to = []

	def clear(self):
		del self.states[:]
		del self.actions_id[:]
		del self.actions_to[:]
		del self.logprobs_id[:]
		del self.logprobs_to[:]
		del self.rewards[:]
		del self.state_values[:]
		del self.is_terminals[:]
		del self.valid_id[:]
		del self.valid_to[:]

class ActorCriticDualHead(nn.Module):
	def __init__(self, state_dim, id_dim, to_dim):
		super().__init__()
		self.base = nn.Sequential(
			nn.Linear(state_dim, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
		)
		self.actor_id = nn.Linear(64, id_dim)
		self.actor_to = nn.Linear(64, to_dim)
		self.critic = nn.Linear(64, 1)

	def forward(self, state):
		x = self.base(state)
		id_logits = self.actor_id(x)
		to_logits = self.actor_to(x)
		value = self.critic(x)
		return id_logits, to_logits, value

	def act(self, state, avail_id=None, avail_to=None, greedy=False):
		id_logits, to_logits, value = self.forward(state)
		if avail_id is not None:
			id_logits = torch.where(avail_id.bool(), id_logits, torch.tensor(-1e8, device=id_logits.device))
		if avail_to is not None:
			to_logits = torch.where(avail_to.bool(), to_logits, torch.tensor(-1e8, device=to_logits.device))
		id_dist = Categorical(logits=id_logits)
		to_dist = Categorical(logits=to_logits)
		if not greedy:
			id_pos = id_dist.sample()
			to_pos = to_dist.sample()
		else:
			id_pos = id_logits.argmax(-1)
			to_pos = to_logits.argmax(-1)
		logprob_id = id_dist.log_prob(id_pos)
		logprob_to = to_dist.log_prob(to_pos)
		return id_pos.detach(), to_pos.detach(), logprob_id.detach(), logprob_to.detach(), value.detach()

	def evaluate(self, state, action_id, action_to):
		id_logits, to_logits, value = self.forward(state)
		id_dist = Categorical(logits=id_logits)
		to_dist = Categorical(logits=to_logits)
		logprob_id = id_dist.log_prob(action_id)
		logprob_to = to_dist.log_prob(action_to)
		entropy_id = id_dist.entropy()
		entropy_to = to_dist.entropy()
		return logprob_id, logprob_to, value, entropy_id, entropy_to

class PPO_DualHead:
	def __init__(self, state_dim, id_dim, to_dim, lr_actor, lr_critic, gamma=0.99, K_epochs=4, eps_clip=0.2):
		self.gamma = gamma
		self.eps_clip = eps_clip
		self.K_epochs = K_epochs

		self.buffer = [RolloutBuffer(), RolloutBuffer()]

		self.policy = ActorCriticDualHead(state_dim, id_dim, to_dim).to(device)
		self.optimizer = torch.optim.Adam([
			{'params': self.policy.base.parameters(), 'lr': lr_actor},
			{'params': self.policy.actor_id.parameters(), 'lr': lr_actor},
			{'params': self.policy.actor_to.parameters(), 'lr': lr_actor},
			{'params': self.policy.critic.parameters(), 'lr': lr_critic}
		])
		self.policy_old = ActorCriticDualHead(state_dim, id_dim, to_dim).to(device)
		self.policy_old.load_state_dict(self.policy.state_dict())
		self.MseLoss = nn.MSELoss()

	def new_buffer(self):
		for buf in self.buffer:
			buf.clear()

	def select_action(self, state, player_id, avail_id=None, avail_to=None, test=False):
		state = np.expand_dims(state, 0)
		state = torch.FloatTensor(state).to(device)
		with torch.no_grad():
			id_pos, to_pos, logprob_id, logprob_to, state_val = self.policy_old.act(
				state, avail_id, avail_to, greedy=test
			)
		if not test:
			self.buffer[player_id].states.append(state)
			self.buffer[player_id].actions_id.append(id_pos)
			self.buffer[player_id].actions_to.append(to_pos)
			self.buffer[player_id].logprobs_id.append(logprob_id)
			self.buffer[player_id].logprobs_to.append(logprob_to)
			self.buffer[player_id].state_values.append(state_val)
			if avail_id is not None:
				self.buffer[player_id].valid_id.append(avail_id.clone().detach())
			if avail_to is not None:
				self.buffer[player_id].valid_to.append(avail_to.clone().detach())
		return int(id_pos.item()), int(to_pos.item())

	def update(self, player_id):
		buf = self.buffer[player_id]
		# 计算回报
		rewards = []
		discounted_reward = 0
		for reward, is_terminal in zip(reversed(buf.rewards), reversed(buf.is_terminals)):
			if is_terminal:
				discounted_reward = 0
			discounted_reward = reward + (self.gamma * discounted_reward)
			rewards.insert(0, discounted_reward)
		rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
		rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

		old_states = torch.cat(buf.states, dim=0).detach()
		old_id = torch.stack(buf.actions_id).detach()
		old_to = torch.stack(buf.actions_to).detach()
		old_logprobs_id = torch.stack(buf.logprobs_id).detach()
		old_logprobs_to = torch.stack(buf.logprobs_to).detach()
		old_state_values = torch.cat(buf.state_values, dim=0).squeeze().detach()

		advantages = rewards.detach() - old_state_values.detach()

		for _ in range(self.K_epochs):
			logprob_id, logprob_to, state_values, entropy_id, entropy_to = self.policy.evaluate(
				old_states, old_id, old_to
			)
			ratios_id = torch.exp(logprob_id - old_logprobs_id)
			ratios_to = torch.exp(logprob_to - old_logprobs_to)
			ratios = ratios_id * ratios_to

			surr1 = ratios * advantages
			surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
			loss_actor = -torch.min(surr1, surr2).mean()
			loss_critic = self.MseLoss(state_values.squeeze(), rewards)
			entropy_loss = -0.01 * (entropy_id.mean() + entropy_to.mean())
			loss = loss_actor + 0.5 * loss_critic + entropy_loss

			self.optimizer.zero_grad()
			loss.backward()
			nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
			self.optimizer.step()

		self.policy_old.load_state_dict(self.policy.state_dict())
		buf.clear()

	def save(self, checkpoint_path):
		torch.save(self.policy_old.state_dict(), checkpoint_path)

	def load(self, checkpoint_path):
		self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
		self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))