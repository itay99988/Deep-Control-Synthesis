import sys
import torch
from arguments import get_args
from ppo import PPO
from network import SingleHeadRNN
from eval_policy import eval_policy
from experiment import *


def experiment_selector(experiment, history_len):
	"""
		initializes the desired experiment
		Parameters:
			experiment - the name of the example
			history_len - nubmer of concatenated short term history vectors
		Return:
			A distributed system object containing both system and environment.
	"""
	if experiment == 'permitted':
		return permitted_experiment_setup(history_len)
	elif experiment == 'schedule':
		return schedule_experiment_setup(history_len)
	elif experiment == 'cases':
		return cases_experiment_setup(history_len)
	elif experiment == 'choice_scc':
		return choice_scc_experiment_setup(history_len)
	elif experiment == 'schedule_cycle':
		return schedule_cycle_experiment_setup(history_len)
	elif experiment == 'cycle_scc':
		return cycle_scc_experiment_setup(history_len)
	else:
		return None


def train(dist_sys, hyperparameters, actor_model, critic_model):
	"""
		Trains the model.
		Parameters:
			env - the environment to train on
			hyperparameters - a dict of hyperparameters to use, defined in main
			actor_model - the actor model to load in if we want to continue training
			critic_model - the critic model to load in if we want to continue training
		Return:
			None
	"""
	print(f"Training", flush=True)

	# Create a model for PPO.
	model = PPO(policy_class=SingleHeadRNN, env=dist_sys, **hyperparameters)

	# Tries to load in an existing actor/critic model to continue training on
	if actor_model != '' and critic_model != '':
		print(f"Loading in {actor_model} and {critic_model}...", flush=True)
		model.actor.load_state_dict(torch.load(actor_model))
		model.critic.load_state_dict(torch.load(critic_model))
		print(f"Successfully loaded.", flush=True)
	elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
		print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
		sys.exit(0)
	else:
		print(f"Training from scratch.", flush=True)

	# Train the PPO model with a specified total timesteps
	# you can kill the process whenever you feel like PPO is converging
	model.learn(total_timesteps=hyperparameters['total_timesteps'])


def test(dist_sys, actor_model, history_len):
	"""
		Tests the model.
		Parameters:
			env - the environment to test the policy on
			actor_model - the actor model to load in
		Return:
			None
	"""
	print(f"Testing {actor_model}", flush=True)

	# If the actor model is not specified, then exit
	if actor_model == '':
		print(f"Didn't specify model file. Exiting.", flush=True)
		sys.exit(0)

	# Extract out dimensions of observation and action spaces
	tr_dim = len(dist_sys.system.transitions)
	input_dim = len(dist_sys.system.states) * tr_dim * history_len

	# Build our policy the same way we build our actor model in PPO
	policy = SingleHeadRNN(input_dim, tr_dim)

	# Load in the actor model saved by the PPO algorithm
	policy.load_state_dict(torch.load(actor_model))

	# Evaluate our policy with a separate module, eval_policy, to demonstrate
	# that once we are done training the model/policy with ppo.py, we no longer need
	# ppo.py since it only contains the training algorithm. The model/policy itself exists
	# independently as a binary file that can be loaded in with torch.
	eval_policy(policy=policy, env=dist_sys)


def main(args):
	"""
		The main function to run.
		Parameters:
			args - the arguments parsed from command line
		Return:
			None
	"""
	# NOTE: Here's where you can set hyperparameters for PPO. I don't include them as part of
	# ArgumentParser because it's too annoying to type them every time at command line. Instead, you can change them here.
	# To see a list of hyperparameters, look in ppo.py at function _init_hyperparameters
	hyperparameters = {
				'history_len': args.history_len,
				'timesteps_per_batch': args.batch_timesteps,
				'max_timesteps_per_episode': args.episode_timesteps,
				'gamma': args.gamma,
				'n_updates_per_iteration': args.iteration_updates,
				'lr': args.lr,
				'clip': args.clip,
				'total_timesteps': args.total_timesteps,
				'initial_temperature': args.initial_temperature,
				'temperature_decay': args.temperature_decay,
				'end_cond_threshold': args.end_cond_threshold
			  }

	print(f"Chosen hyperparameters - '{args.experiment}' experiment")
	print(hyperparameters)

	# Creates the environment we will be running
	dist_sys = experiment_selector(args.experiment, args.history_len)
	if not dist_sys:
		print(f"Didn't specify a valid experiment name. Exiting.", flush=True)
		sys.exit(0)

	# Train or test, depending on the mode specified
	if args.mode == 'train':
		train(dist_sys=dist_sys, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)
	else:
		test(dist_sys=dist_sys, actor_model=args.actor_model, history_len=args.history_len)


if __name__ == '__main__':
	args = get_args()  # Parse arguments from command line
	main(args)
