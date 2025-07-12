import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

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


################################## PPO Policy ##################################
class RolloutBuffer:
	def __init__(self):
		self.actions = []
		self.states = []
		self.logprobs = []
		self.rewards = []
		self.state_values = []
		self.is_terminals = []
	
	def clear(self):
		del self.actions[:]
		del self.states[:]
		del self.logprobs[:]
		del self.rewards[:]
		del self.state_values[:]
		del self.is_terminals[:]


#
# ========================================================================================
# 步骤一：用下面的全新代码替换文件中原有的 ActorCritic 类
# ========================================================================================
#

class ActorCritic(nn.Module):
	def __init__(self, state_dim, action_dim_piece, action_dim_move):
		super(ActorCritic, self).__init__()

		# --- 核心组件定义 ---

		hidden_dim = 64  # 您可以根据需要调整隐藏层维度
		piece_embedding_dim = 16 # 将棋子ID转换为一个更丰富的向量表示

		# 1. 共享主体 (Shared Body): 负责从状态中提取通用特征
		#    对于初步研究，MLP足够了。未来可以换成CNN。
		self.body = nn.Sequential(
			nn.Linear(state_dim, hidden_dim),
			nn.Tanh()
		)

		# 2. 棋子嵌入层: 这是实现依赖关系的关键。它将第一阶段选择的棋子ID转换为一个向量。
		self.piece_embed = nn.Embedding(action_dim_piece, piece_embedding_dim)

		# 3. 演员头 (Actor Heads)
		#    a. 棋子选择头: 接收通用特征，决定选择哪个棋子
		self.actor_piece = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.Tanh(),
			nn.Linear(hidden_dim, action_dim_piece)  # 输出原始 logits
		)
		
		#    b. 走法选择头: 接收通用特征 和 棋子嵌入，决定如何移动
		#       输入维度是 hidden_dim (来自状态) + piece_embedding_dim (来自选择的棋子)
		self.actor_move = nn.Sequential(
			nn.Linear(hidden_dim + piece_embedding_dim, hidden_dim),
			nn.Tanh(),
			nn.Linear(hidden_dim, action_dim_move)  # 输出原始 logits
		)

		# 4. 批评家头 (Critic Head): 评估状态价值，保持不变
		self.critic = nn.Sequential(
			nn.Linear(state_dim, hidden_dim),
			nn.Tanh(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.Tanh(),
			nn.Linear(hidden_dim, 1)
		)

	def act(self, player, state, piece_mask, move_mask_func):
		"""
		在与环境交互时，执行一个完整的两阶段动作。
		这个方法处理了所有依赖关系。
		- state: 当前环境状态
		- piece_mask: 当前合法的棋子选择掩码
		- move_mask_func: 一个函数，调用它并传入选择的棋子ID，能返回对应的合法走法掩码
		"""
		# --- Phase 1: 选择棋子 ---
		shared_features = self.body(state)
		
		# 应用掩码
		piece_logits = self.actor_piece(shared_features)
		if piece_mask is not None:
			piece_logits[piece_mask == 0] = -1e8
		
		dist_piece = Categorical(logits=piece_logits)
		action_piece = dist_piece.sample()
		logprob_piece = dist_piece.log_prob(action_piece)

		# --- Phase 2: 选择走法 (依赖于 Phase 1 的结果) ---
		piece_embedding = self.piece_embed(action_piece)
		
		# 将状态特征和棋子嵌入拼接，形成决策依赖
		move_input = torch.cat([shared_features, piece_embedding], dim=-1)

		# 获取依赖于 action_piece 的走法掩码
		move_mask = move_mask_func(player, action_piece.item()).flatten()
		move_mask_tensor = torch.FloatTensor(move_mask).to(state.device)

		move_logits = self.actor_move(move_input)
		if move_mask_tensor is not None:
			move_logits[move_mask_tensor == 0] = -1e8
		
		dist_move = Categorical(logits=move_logits)
		action_move = dist_move.sample()
		logprob_move = dist_move.log_prob(action_move)
		
		# --- 组合结果 ---
		# 动作是一个包含两个值的张量
		action = torch.stack([action_piece, action_move])
		# 总对数概率是两者之和
		action_logprob = logprob_piece + logprob_move
		# 价值只与初始状态有关
		state_val = self.critic(state)

		return action.detach(), action_logprob.detach(), state_val.detach()
	
	def evaluate(self, state, action):
		"""
		在更新网络时，重新评估一个批次的旧动作。
		- state: 形状为 [batch_size, state_dim]
		- action: 形状为 [batch_size, 2]
		"""
		# --- 重新评估 Phase 1: 棋子选择 ---
		# 拆分复合动作
		action_piece = action[:, 0].long()
		action_move = action[:, 1].long()

		shared_features = self.body(state)
		
		piece_logits = self.actor_piece(shared_features)
		dist_piece = Categorical(logits=piece_logits)
		logprob_piece = dist_piece.log_prob(action_piece)
		entropy_piece = dist_piece.entropy()

		# --- 重新评估 Phase 2: 走法选择 ---
		piece_embedding = self.piece_embed(action_piece)
		move_input = torch.cat([shared_features, piece_embedding], dim=-1)

		move_logits = self.actor_move(move_input)
		dist_move = Categorical(logits=move_logits)
		logprob_move = dist_move.log_prob(action_move)
		entropy_move = dist_move.entropy()

		# --- 组合结果 ---
		# 总对数概率
		action_logprobs = logprob_piece + logprob_move
		# 总熵
		dist_entropy = entropy_piece + entropy_move
		# 状态价值
		state_values = self.critic(state)
		
		return action_logprobs, state_values, dist_entropy

class PPO:
	# 在 PPO.__init__ 中
# state_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip 保持不变
# action_dim 需要换成两个
	def __init__(self, state_dim, action_dim_piece, action_dim_move, lr_actor, lr_critic, gamma, K_epochs, eps_clip):

		self.gamma = gamma
		self.eps_clip = eps_clip
		self.K_epochs = K_epochs
		self.has_continuous_action_space = False

		self.buffer = [RolloutBuffer(), RolloutBuffer()]

		# 使用新的方式实例化
		self.policy = ActorCritic(state_dim, action_dim_piece, action_dim_move).to(device)
		self.optimizer = torch.optim.Adam([
						{'params': self.policy.body.parameters()}, # 共享部分
						{'params': self.policy.critic.parameters(), 'lr': lr_critic},
						{'params': self.policy.actor_piece.parameters(), 'lr': lr_actor},
						{'params': self.policy.actor_move.parameters(), 'lr': lr_actor}
					])

		self.policy_old = ActorCritic(state_dim, action_dim_piece, action_dim_move).to(device)
		self.policy_old.load_state_dict(self.policy.state_dict())

		self.MseLoss = nn.MSELoss()

	def set_action_std(self, new_action_std):
		if self.has_continuous_action_space:
			self.action_std = new_action_std
			self.policy.set_action_std(new_action_std)
			self.policy_old.set_action_std(new_action_std)
		else:
			print("--------------------------------------------------------------------------------------------")
			print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
			print("--------------------------------------------------------------------------------------------")

	def decay_action_std(self, action_std_decay_rate, min_action_std):
		print("--------------------------------------------------------------------------------------------")
		if self.has_continuous_action_space:
			self.action_std = self.action_std - action_std_decay_rate
			self.action_std = round(self.action_std, 4)
			if (self.action_std <= min_action_std):
				self.action_std = min_action_std
				print("setting actor output action_std to min_action_std : ", self.action_std)
			else:
				print("setting actor output action_std to : ", self.action_std)
			self.set_action_std(self.action_std)

		else:
			print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
		print("--------------------------------------------------------------------------------------------")

	def select_action(self, player, state, piece_mask, move_mask_func): # 需要传入 env 以获取掩码
		with torch.no_grad():
			state_tensor = torch.FloatTensor(state).to(device)
			
			# 从环境中获取初始的棋子掩码
			# piece_mask = env.get_piece_mask() # @ LTY 此处修改同下
			piece_mask_tensor = torch.FloatTensor(piece_mask).to(device)

			# 从环境中获取一个“函数”，用于根据棋子ID返回移动掩码
			# 这是处理依赖的关键
			# move_mask_func = env.get_move_mask # @LTY 这里需要修改成 env 里面对应的函数，但是直接传一个 Environment 会不会复杂度太高了？这是一个问题。修改方案为把 select_action 拆成两部分
			
			# 调用我们重写后的 act 方法
			action, action_logprob, state_val = self.policy_old.act(player, state_tensor, piece_mask_tensor, move_mask_func)
		
		# 存储数据到 buffer
		self.buffer[player].states.append(state_tensor)
		self.buffer[player].actions.append(action) # action 是一个 (2,) 的张量
		self.buffer[player].logprobs.append(action_logprob)
		self.buffer[player].state_values.append(state_val)

		return action.cpu().numpy() # 返回一个 [piece_id, move_pos] 的 numpy 数组
	
	def update(self, player):
	# ... (计算 rewards 的部分保持不变) ...
		rewards = []
		discounted_reward = 0
		for reward, is_terminal in zip(reversed(self.buffer[player].rewards), reversed(self.buffer[player].is_terminals)):
			if is_terminal:
				discounted_reward = 0
			discounted_reward = reward + (self.gamma * discounted_reward)
			rewards.insert(0, discounted_reward)
		   
	# Normalizing the rewards (保持不变)
		rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
		rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

		# convert list to tensor
		old_states = torch.squeeze(torch.stack(self.buffer[player].states, dim=0)).detach().to(device)
		# !!! 这里的 action 现在是 [batch_size, 2] 的形状 !!!
		old_actions = torch.stack(self.buffer[player].actions, dim=0).detach().to(device)
		old_logprobs = torch.squeeze(torch.stack(self.buffer[player].logprobs, dim=0)).detach().to(device)
		old_state_values = torch.squeeze(torch.stack(self.buffer[player].state_values, dim=0)).detach().to(device)

		# calculate advantages (保持不变)
		advantages = rewards.detach() - old_state_values.detach()

		# Optimize policy for K epochs
		for _ in range(self.K_epochs):
			# Evaluating old actions and values
			# 调用我们重写的 evaluate，它会返回合并后的 logprobs 和 entropy
			logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)


			# match state_values tensor dimensions with rewards tensor
			state_values = torch.squeeze(state_values)
			
			# Finding the ratio (pi_theta / pi_theta__old)
			ratios = torch.exp(logprobs - old_logprobs.detach())

			# Finding Surrogate Loss  
			surr1 = ratios * advantages
			surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

			# final loss of clipped objective PPO
			loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
			
			# take gradient step
			self.optimizer.zero_grad()
			loss.mean().backward()
			self.optimizer.step()
			
		# Copy new weights into old policy
		self.policy_old.load_state_dict(self.policy.state_dict())

		# clear buffer
		self.buffer[player].clear()
	
	def save(self, checkpoint_path):
		torch.save(self.policy_old.state_dict(), checkpoint_path)
   
	def load(self, checkpoint_path):
		self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
		self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
