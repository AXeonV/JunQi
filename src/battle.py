import os
import glob
import time
from datetime import datetime

import torch
import numpy as np
import time

from DH_PPO.model import PPO
from JunQi.JunQi import JunqiEnv
import warnings
warnings.filterwarnings("ignore")

#################################### Testing ###################################
def battle():
	####### initialize environment hyperparameters ######
	has_continuous_action_space = False # continuous action space; else discrete

	max_ep_len = 1000                   # max timesteps in one episode
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
	state_dim = [4441, 4441]
	action_id_dim = 25
	action_to_dim = 60
	print("testing environment name : JunQi")
	
	timestamp = datetime.now().strftime('%Y%m%d.%H%M%S')

	print("============================================================================================")
	agent = [
	PPO(state_dim[0], action_id_dim, action_to_dim, lr_actor, lr_critic, 0.99, 4, 0.2),
	PPO(state_dim[1], action_id_dim, action_to_dim, lr_actor, lr_critic, 0.99, 4, 0.2)
  ]

	directory = "data/"
	checkpoint_path0 = directory + "PPO_JunQi_0_4_0.pth"
	checkpoint_path1 = directory + "PPO_JunQi_0_2_0.pth"
	print("loading network0 from : " + checkpoint_path0)
	print("loading network1 from : " + checkpoint_path1)
	agent[0].load(checkpoint_path0)
	agent[1].load(checkpoint_path1)
	print("--------------------------------------------------------------------------------------------")

	mx_history = [50, 50]
	total_test_episodes = 100
	win = [0, 0]
	sssp = 0
	for ep in range(1, total_test_episodes+1):
   
		print("Episode: {}".format(ep))
		env.reset()

		for t in range(1, max_ep_len+1):
						
			sssp += 2
			done = False
			winner = -1
	 
			for i in range(2):
				winner = i
				avail_id = env._get_selection_mask(i).flatten()
				if np.all(avail_id == 0):
					winner = 1 - i
					done = True
					break
				state = env.extract_state(i, 0)
				action_id, action_to = agent[i].select_action(i, state, avail_id, env._get_movement_mask)
				reward, done = env.Tstep(i, action_id, action_to)
				if done:
					break
			
			if done:
				win[winner] += 1
				break

	env.close()
	print(sssp / total_test_episodes)
	print("N vs P \t\t Win: {}% \t\t Lose: {}%".format(round(win[0] / total_test_episodes * 100, 2), round(win[1] / total_test_episodes * 100, 2)))

	print("============================================================================================")


if __name__ == '__main__':
	battle()
