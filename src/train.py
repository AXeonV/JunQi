import os
import time
from datetime import datetime
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

from JunQi.JunQi import JunqiEnv  # 假设军棋环境在此模块中
from Nash.model import Nash       # 假设PPO实现类

def train():
	####### initialize environment hyperparameters ######
	has_continuous_action_space = False # continuous action space; else discrete

	max_ep_len = 1000                   # max timesteps in one episode
	max_training_timesteps = int(1e6)   # break training loop if timeteps > max_training_timesteps

	print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
	log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
	save_model_freq = int(5000)         # save model frequency (in num timesteps)

	action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
	action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
	min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
	action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
	eval_num = 25
	last_save_model_step = 0
	max_win_ratio = 0
	#####################################################

	#####################################################
	n = 0
	delta_m = 20
	m = 0

	## Note : print/log frequencies should be > than max_ep_len

	################ PPO hyperparameters ################
	update_timestep = max_ep_len * 4      # update policy every n timesteps

	lr_actor = 0.004       # learning rate for actor network
	lr_critic = 0.004       # learning rate for critic network

	random_seed = 0         # set random seed if required (0 = no random seed)
	#####################################################
	# 初始化环境
	env = JunqiEnv()
	'''
	state_dim = {
		'Pri_I': (5, 12, 12),
		'Pub_oppo': (5, 12, 12),
		'Move': (5, 12, 25),
		'phase': (1,),
		'selected': (5, 12),
		'steps_since_attack': (1,),
		'isMoved_I': (25,)
	}
	'''
	state_dim = 2582
	action_dim = 5 * 12  # 最大动作空间
	print("training environment name : JunQi")
	env_name = "JunQi"
	###################### logging ######################

	#### log files for multiple runs are NOT overwritten
	log_dir = "logs"
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

	log_dir = log_dir + '/'
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

	#### get number of log files in log directory
	run_num = 0
	current_num_files = next(os.walk(log_dir))[2]
	run_num = len(current_num_files)

	#### create new log file for each run
	log_f_name = log_dir + '/Nash_' + env_name + "_log_" + str(run_num) + ".csv"

	print("current logging run number for " + env_name + " : ", run_num)
	print("logging at : " + log_f_name)
	#####################################################

	################### checkpointing ###################
	# directory = "data/"
	# checkpoint_path = directory + "Nash_{}_{}_{}_0.pth".format(env_name, 0, 1)
	# print("loading network from : " + checkpoint_path)

	run_num_pretrained = 3      #### change this to prevent overwriting weights in same env_name folder

	directory = "data"
	if not os.path.exists(directory):
		os.makedirs(directory)

	directory = directory + '/'
	if not os.path.exists(directory):
		os.makedirs(directory)


	checkpoint_path = directory + "Nash_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
	print("save checkpoint path : " + checkpoint_path)
	#####################################################


	############# print all hyperparameters #############
	print("--------------------------------------------------------------------------------------------")
	print("max training timesteps : ", max_training_timesteps)
	print("max timesteps per episode : ", max_ep_len)
	print("model saving frequency : " + str(save_model_freq) + " timesteps")
	print("log frequency : " + str(log_freq) + " timesteps")
	print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
	print("--------------------------------------------------------------------------------------------")
	print("state space dimension : ", state_dim)
	print("action space dimension : ", action_dim)
	print("--------------------------------------------------------------------------------------------")
	if has_continuous_action_space:
		print("Initializing a continuous action space policy")
		print("--------------------------------------------------------------------------------------------")
		print("starting std of action distribution : ", action_std)
		print("decay rate of std of action distribution : ", action_std_decay_rate)
		print("minimum std of action distribution : ", min_action_std)
		print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
	else:
		print("Initializing a discrete action space policy")
	print("--------------------------------------------------------------------------------------------")
	print("update frequency : " + str(update_timestep) + " timesteps")
	print("--------------------------------------------------------------------------------------------")
	print("optimizer learning rate actor : ", lr_actor)
	print("optimizer learning rate critic : ", lr_critic)
	if random_seed:
		print("--------------------------------------------------------------------------------------------")
		print("setting random seed to ", random_seed)
		torch.manual_seed(random_seed)
  	# env.seed(random_seed)
		print("JunQi env do not support random seed setting, so it is not set")
		np.random.seed(random_seed)
	#####################################################

	print("============================================================================================")

	################# training procedure ################
	# 初始化PPO
	nash_agent = Nash(state_dim, action_dim, lr_actor, lr_critic, has_continuous_action_space, action_std, flatten=True)

	# loading pretrained model(if needed)
	load_checkpoint = True
	if load_checkpoint:
		load_checkpoint_path = directory + "Nash_JunQi_0_2_0.pth"
		print("loading network from : " + load_checkpoint_path)
		if os.path.exists(load_checkpoint_path):
			nash_agent.load(load_checkpoint_path)
		else:
			print("checkpoint not found at : " + load_checkpoint_path)
			exit(0)

	# track total training time
	start_time = datetime.now().replace(microsecond=0)
	print("Started training at (GMT) : ", start_time)

	print("============================================================================================")

	# logging file
	log_f = open(log_f_name,"w+")
	log_f.write('episode,timestep,reward\n')

	# printing and logging variables
	print_running_reward = 0
	print_nash_running_reward = 0
	print_running_episodes = 0

	log_running_reward = 0
	log_running_episodes = 0

	time_step = 0
	i_episode = 0

	# training loop
	sete = 0
	chosen_wrong_reward = -100
	while time_step < max_training_timesteps:
		env.reset()
		current_ep_reward = 0
		current_nash_ep_reward = 0
		nash_agent.new_buffer()
		done = False
		rewards = []

		for t in range(1, max_ep_len+1):
			slection_mask = np.zeros(60, dtype=np.int16)
			for i in range(2):
				sete += 1
				avail_actions0 = env.get_onehot_available_actions(0, i, 0)
    
				# 另外终止情况1：
				if np.all(avail_actions0 == 0):
					done = True
					break
				state = env.extract_state(i, 0, slection_mask)
				action0 = nash_agent.select_action(state, i, avail_actions0, test=True)
				avail_actions1 = env.get_onehot_available_actions(1, i, action0)

				# 另外终止情况2：
				while np.all(avail_actions1 == 0):
					avail_actions0[action0] = 0
					if np.all(avail_actions0 == 0):
						done = True
						break
					nash_agent.buffer[i].rewards[-1].append(chosen_wrong_reward)
					nash_agent.buffer[i].is_terminals[-1].append(done)
					action0 = nash_agent.select_action(state, i, avail_actions0, test=True)
					avail_actions1 = env.get_onehot_available_actions(1, i, action0)
     
				if done:
					nash_agent.buffer[i].rewards[-1].append(chosen_wrong_reward)
					nash_agent.buffer[i].is_terminals[-1].append(done)
					break
   
				slection_mask[action0] = 1
				state = env.extract_state(i, 1, slection_mask)
				action1 = nash_agent.select_action(state, i, avail_actions1, test=True)
				reward, done = env.Tstep(i, action0, action1)
    
				rewards.append(reward)
				nash_agent.buffer[i].rewards[-1].append(reward)
				nash_agent.buffer[i].is_terminals[-1].append(done)
				nash_agent.buffer[i].rewards[-1].append(reward)
				nash_agent.buffer[i].is_terminals[-1].append(done)
    
			time_step += 1
			current_ep_reward += np.abs(rewards[0] + rewards[1])
			if rewards[0] != 0:
				current_nash_ep_reward = current_nash_ep_reward + rewards[0]
			else:
				current_nash_ep_reward = current_nash_ep_reward - rewards[1]
			if time_step % log_freq == 0:

				# log average reward till last episode
				log_avg_reward = log_running_reward / log_running_episodes
				log_avg_reward = round(log_avg_reward, 4)

				log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
				log_f.flush()

				log_running_reward = 0
				log_running_episodes = 0

			# printing average reward
			if time_step % print_freq == 0:

				# print average reward till last episode
				print_avg_reward = print_running_reward / print_running_episodes
				print_avg_nash_reward = print_nash_running_reward / print_running_episodes
				print_avg_reward = round(print_avg_reward, 2)
				print_avg_nash_reward = round(print_avg_nash_reward, 2)

				print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {} \t\t Nash Reward : {}".format(i_episode, time_step, print_avg_reward, print_avg_nash_reward))

				print_running_reward = 0
				print_nash_running_reward = 0
				print_running_episodes = 0

			# break if the episode is over
			if done:
				# save model weights
				if time_step - last_save_model_step > save_model_freq:
					print("--------------------------------------------------------------------------------------------")
					print("saving model at : " + checkpoint_path)
					nash_agent.save(checkpoint_path[:-4] + "_" + str(max_win_ratio) + checkpoint_path[-4:])
					print("model saved")
					print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
					print("M: {}".format(m))
					print("--------------------------------------------------------------------------------------------")
					last_save_model_step = time_step
				break
  
		# update PPO agent
		if time_step % update_timestep == 0:
			if n >= delta_m:
				n = 0
				m += 1
				nash_agent.policy_reg_old.load_state_dict(nash_agent.policy_reg.state_dict())
				nash_agent.policy_reg.load_state_dict(nash_agent.policy_old.state_dict())
			alpha = 1 if n > delta_m / 2 else n * 2 / delta_m
			nash_agent.update(alpha)
			n += 1

		print_running_reward += current_ep_reward
		print_running_episodes += 1

		print_nash_running_reward += current_nash_ep_reward

		log_running_reward += current_ep_reward
		log_running_episodes += 1

		i_episode += 1



if __name__ == "__main__":
	train()