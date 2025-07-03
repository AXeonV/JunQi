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
	state_dim = 2582
	action_dim = 5 * 12  # 最大动作空间
	print("testing environment name : JunQi")
	
	timestamp = datetime.now().strftime('%Y%m%d.%H%M%S')

	print("============================================================================================")
	nash_agent = [
	Nash(state_dim, action_dim, lr_actor, lr_critic, has_continuous_action_space, action_std, flatten=True)
	,Nash(state_dim, action_dim, lr_actor, lr_critic, has_continuous_action_space, action_std, flatten=True)
  ]

	directory = "data/"
	checkpoint_path0 = directory + "Nash_JunQi_0_5_0.pth"
	checkpoint_path1 = directory + "Nash_JunQi_0_1_0.pth"
	print("loading network0 from : " + checkpoint_path0)
	print("loading network1 from : " + checkpoint_path1)
	nash_agent[0].load(checkpoint_path0)
	nash_agent[1].load(checkpoint_path1)
	print("--------------------------------------------------------------------------------------------")

	total_test_episodes = 100
	win = [0, 0]
	sssp = 0
	for ep in range(1, total_test_episodes+1):
		
		env.reset()

		for t in range(1, max_ep_len+1):
			slection_mask = np.zeros(60, dtype=np.int16)

			# from JunQi.wboard import print_state
			# board_in = env.output()
			# board_out = []
			# for i in range(12):
			# 	for j in range(5):
			# 		if board_in[0][i][j][1] == 1:
			# 			board_out.append((i, j, (255, 0, 0), board_in[0][i][j][0]))
			# 		elif board_in[0][i][j][1] == -1:
			# 			board_out.append((i, j, (0, 0, 0), board_in[0][i][j][0]))
			# print_state(board_out, int(sssp / 2), timestamp, board_in[1])
						
			sssp += 2
			done = False
	 
			for i in range(2):
				winner = i
				avail_actions0 = env.get_onehot_available_actions(0, i, 0)
				# 另外终止情况1：
				if np.all(avail_actions0 == 0):
					winner = 1 - i
					win[winner] += 1
					done = True
					break
				state = env.extract_state(i, 0, slection_mask)
				
				action0 = nash_agent[i].select_action(state, i, avail_actions0, test=True)
				avail_actions1 = env.get_onehot_available_actions(1, i, action0)
				# 另外终止情况2：
				while np.all(avail_actions1 == 0):
					avail_actions0[action0] = 0
					if np.all(avail_actions0 == 0):
						done = True
						break
					action0 = nash_agent[i].select_action(state, i, avail_actions0, test=True)
					avail_actions1 = env.get_onehot_available_actions(1, i, action0)
				if done:
					winner = 1 - i
					win[winner] += 1
					break
				slection_mask[action0] = 1
				state = env.extract_state(i, 1, slection_mask)
				action1 = nash_agent[i].select_action(state, i, avail_actions1, test=True)
				reward, done = env.Tstep(i, action0, action1)
				
				if done:
					win[winner] += 1
					break
			
			if done:
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
