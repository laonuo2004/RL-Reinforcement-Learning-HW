# enconding: utf-8

env_name 			= "CarRacing-v0"
action_name 		= ["left", "right", "accelerate", "brake", 
						"left acc", "right acc", "left brake", "right brake",
						"nothing"]
max_episode_length 	= 1000000

output_size 		= 9
seed 				= 1

lr 					= 1e-4
gamma 				= 0.99
tau 				= 1.0
entropy_coef		= 0.01
value_loss_coef 	= 0.5
max_grad_norm 		= 50

num_steps 			= 32		# batch size
num_processes 		= 0

test_interval		= 60

#在show.py中使用
show_rewards_curve	= False
show_action			= True

# 在main.py中使用
load_model 			= False
