"""
	This file contains the arguments to parse at command line.
	File ppo_experiment.py will call get_args, which then the arguments
	will be returned.
"""
import argparse


def get_args():
    """
		Description:
		Parses arguments at command line.
		Parameters:
			None
		Return:
			args - the arguments parsed
	"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', dest='mode', type=str, default='train')  # can be 'train' or 'test'
    parser.add_argument('--actor_model', dest='actor_model', type=str, default='')  # your actor model filename
    parser.add_argument('--critic_model', dest='critic_model', type=str, default='')  # your critic model filename
    parser.add_argument('--experiment', dest='experiment', type=str, default='permitted')  # experiment name
    parser.add_argument('--history', dest='history_len', type=int, default='1')  # history length
    parser.add_argument('--batch_timesteps', dest='batch_timesteps', type=int, default='4000')  # allowed timesteps for each batch
    parser.add_argument('--episode_timesteps', dest='episode_timesteps', type=int, default='25')  # timesteps foreach episode
    parser.add_argument('--gamma', dest='gamma', type=float, default='0.99')  # discount factor
    parser.add_argument('--iteration_updates', dest='iteration_updates', type=int, default='10')  # iteration updates foreach batch
    parser.add_argument('--lr', dest='lr', type=float, default='1e-2')  # learning rate
    parser.add_argument('--clip', dest='clip', type=float, default='0.2')  # clipping epsilon
    parser.add_argument('--total_timesteps', dest='total_timesteps', type=int, default='1500000')  # total timesteps
    parser.add_argument('--init_tem', dest='initial_temperature', type=float, default='2')  # initial temperature
    parser.add_argument('--tem_decay', dest='temperature_decay', type=float, default='1.008')  #temperature decay factor
    parser.add_argument('--end_cond', dest='end_cond_threshold', type=float, default='0.99')  # end condition percentage

    args = parser.parse_args()

    return args
