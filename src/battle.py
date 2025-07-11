import os
import glob
import time
from datetime import datetime

import torch
import numpy as np
import time

from Nash.model import Nash
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
	action_dim = 25 * 60
	print("testing environment name : JunQi")
	
	timestamp = datetime.now().strftime('%Y%m%d.%H%M%S')

	print("============================================================================================")
	nash_agent = [
	Nash(state_dim[0], action_dim, lr_actor, lr_critic, has_continuous_action_space, action_std, flatten=True),
	Nash(state_dim[1], action_dim, lr_actor, lr_critic, has_continuous_action_space, action_std, flatten=True)
  ]

	directory = "data/"
	checkpoint_path0 = directory + "Nash_JunQi_0_6_0.pth"
	checkpoint_path1 = directory + "Nash_JunQi_0_2_0.pth"
	print("loading network0 from : " + checkpoint_path0)
	print("loading network1 from : " + checkpoint_path1)
	nash_agent[0].load(checkpoint_path0)
	nash_agent[1].load(checkpoint_path1)
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
				avail_actions = env.get_onehot_available_actions(i)
				if np.all(avail_actions == 0):
					winner = 1 - i
					done = True
					break
				state = env.extract_state(i, mx_history[i])
				action = nash_agent[i].select_action(state, i, avail_actions, test=True)
				reward, done = env.Tstep(i, action)
				
				if done:
					break
			
			if done:
				win[winner] += 1
				break

		# clear buffer
		nash_agent[0].buffer.clear()
		nash_agent[1].buffer.clear()
	env.close()
	print(sssp / total_test_episodes)
	print("N vs P \t\t Win: {}% \t\t Lose: {}%".format(round(win[0] / total_test_episodes * 100, 2), round(win[1] / total_test_episodes * 100, 2)))

	print("============================================================================================")


if __name__ == '__main__':
	battle()
