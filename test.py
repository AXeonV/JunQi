import os
import glob
import time
from datetime import datetime

import torch
import numpy as np
import time

from model import Nash
from JunQi import JunqiEnv
import warnings
warnings.filterwarnings("ignore")

#################################### Testing ###################################
def test():
	####### initialize environment hyperparameters ######
	has_continuous_action_space = False # continuous action space; else discrete

	max_ep_len = 20                     # max timesteps in one episode
	max_training_timesteps = int(3e4)   # break training loop if timeteps > max_training_timesteps

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
	update_timestep = max_ep_len * 4   # update policy every n timesteps

	lr_actor = 0.0004                  # learning rate for actor network
	lr_critic = 0.0004                 # learning rate for critic network

	random_seed = 0                    # set random seed if required (0 = no random seed)
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
	print("testing environment name : JunQi")
	env_name = "JunQi"
	

	print("============================================================================================")
	nash_agent = Nash(state_dim, action_dim, lr_actor, lr_critic, has_continuous_action_space, action_std, flatten=True)

	random_seed = 0             #### set this to load a particular checkpoint trained on random seed
	run_num_pretrained = 2      #### set this to load a particular checkpoint num

	directory = "pth" + '/' + env_name + '/'
	checkpoint_path = directory + "Nash_{}_{}_{}_0.pth".format(env_name, random_seed, run_num_pretrained)
	print("loading network from : " + checkpoint_path)

	nash_agent.load(checkpoint_path)

	print("--------------------------------------------------------------------------------------------")

	test_running_reward = 0
	total_test_episodes = 1
	win = [0, 0]
	sssp = 0
	for ep in range(1, total_test_episodes+1):
		
		env.reset()

		for t in range(1, max_ep_len+1):
			slection_mask = np.zeros(60, dtype=np.int16)
			
			if sssp % 1 == 0:
				print(env.output())
				print()
			sssp += 1
			
			done = False
			for i in range(2):
				avail_actions = env.get_onehot_available_actions(0, i, 0)
				state = env.extract_state(i, 0, slection_mask)
				action0 = nash_agent.select_action(state, i, avail_actions, test=True)
				avail_actions = env.get_onehot_available_actions(1, i, action0)
				slection_mask[action0] = 1
				state = env.extract_state(i, 1, slection_mask)
				action1 = nash_agent.select_action(state, i, avail_actions, test=True)
				reward, done = env.Tstep(i, action0, action1)
					
				# print(env.index_to_pos(action0), env.index_to_pos(action1))
				if done:
					win[i] += 1
					break
				
			if done:
				sssp += t
				break

		# clear buffer
		nash_agent.buffer.clear()
	env.close()
	print(sssp / total_test_episodes)

	avg_test_reward = test_running_reward / total_test_episodes
	avg_test_reward = round(avg_test_reward, 2)
	print("average test pure reward : " + str(avg_test_reward))
	print("Nash vs Oppo \t\t Win: {}% \t\t Lose: {}%".format(round(win[0] / total_test_episodes * 100, 2), round((total_test_episodes - win[0]) / total_test_episodes * 100, 2)))

	print("============================================================================================")


if __name__ == '__main__':
	test()
