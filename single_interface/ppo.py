import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.categorical import Categorical


class PPO:
    """
		This is the PPO class we will use as our model in main.py
	"""

    def __init__(self, policy_class, env, **hyperparameters):
        """
			Initializes the PPO model, including hyperparameters.
			Parameters:
				policy_class - the policy class to use for our actor/critic networks.
				env - the environment to train on.
				hyperparameters - all extra arguments passed into PPO that should be hyperparameters.
			Returns:
				None
		"""

        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(hyperparameters)

        # Extract environment information
        self.env = env
        self.tr_dim = len(env.system.transitions)
        self.input_dim = len(env.system.states) * self.tr_dim * self.history_len
        self.end_condition = self.end_cond_threshold * self.max_timesteps_per_episode

        # Initialize actor and critic networks
        self.actor = policy_class(self.input_dim, self.tr_dim)  # ALG STEP 1
        self.critic = policy_class(self.input_dim, 1)

        # Initialize optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time(),
            't_so_far': 0,  # timesteps so far
            'i_so_far': 0,  # iterations so far
            'batch_lens': [],  # episodic lengths in batch
            'batch_rews': [],  # episodic returns in batch
            'actor_losses': [],  # losses of actor network in current iteration
        }

    def learn(self, total_timesteps):
        """
			Parameters:
				total_timesteps - the total number of timesteps to train for
			Return:
				None
		"""
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Iterations ran so far
        avg_ep_return_lst = []   # last average episodic returns

        while t_so_far < total_timesteps:  # ALG STEP 2
            self.initial_temperature /= self.temperature_decay

            # Here we're collecting our batch simulations
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()  # ALG STEP 3

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # convert batch_obs to rnn batch
            batch_size = int(self.timesteps_per_batch / self.max_timesteps_per_episode)
            batch_obs = batch_obs.view(batch_size, self.max_timesteps_per_episode, -1)

            # Calculate advantage at k-th iteration
            V, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()  # ALG STEP 5

            # One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
            # isn't theoretically necessary, but in practice it decreases the variance of
            # our advantages and makes convergence much more stable and faster. I added this because
            # solving some environments was too unstable without it.
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # This is the loop where we update our network for some n epochs
            for _ in range(self.n_updates_per_iteration):  # ALG STEP 6 & 7
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                # NOTE: we just subtract the logs, which is the same as
                # dividing the values and then canceling the log with e^log.
                # For why we use log probabilities instead of actual probabilities,
                # here's a great explanation:
                # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
                # TL;DR makes gradient ascent easier behind the scenes.
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses.
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # Calculate actor and critic losses.
                # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
                # the performance function, but Adam minimizes the loss. So minimizing the negative
                # performance function maximizes it.
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Log actor loss
                self.logger['actor_losses'].append(actor_loss.detach())

            # end condition
            last_ear = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
            avg_ep_return_lst.append(last_ear)
            avg_ep_return = np.mean(avg_ep_return_lst[-8:])
            if avg_ep_return >= self.end_condition:
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')
                break

            # Print a summary of our training so far
            self._log_summary()

            # Save our model if it's time
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')

    def rollout(self):
        """
				None
			Return:
				batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
				batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
				batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
				batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
				batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
		"""
        # Batch data. For more details, check function header.
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []

        t = 0  # Keeps track of how many timesteps we've run so far this batch

        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < self.timesteps_per_batch:
            ep_rews = []  # rewards collected per episode

            # Reset the environment. sNote that obs is short for observation.
            self.env.reset()
            state = self.env.get_compound_state()
            actor_hidden_state = self.actor.init_hidden()

            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            for ep_t in range(self.max_timesteps_per_episode):

                t += 1  # Increment timesteps ran this batch so far

                # Track observations in this batch
                batch_obs.append(state)

                # Calculate action and make a step in the env.
                # Note that rew is short for reward.
                action, log_prob, actor_hidden_state, fil_logits = self.get_action(state, actor_hidden_state)
                state, reward = self.env.step(action)

                # Track recent reward, action, and action log probability
                ep_rews.append(reward)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                # if t == 1:
                #     print(F.softmax(fil_logits, dim=0))

            # Track episodic lengths and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)  # ALG STEP 4

        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews):
        """
			Compute the Reward-To-Go of each timestep in a batch given the rewards.
			Parameters:
				batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)
			Return:
				batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
		"""
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0  # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def get_action(self, state, actor_hidden_state):
        """
			Queries an action from the actor network, should be called from rollout.
			Parameters:
				state - the observation at the current timestep
			Return:
				action - the action to take, as a numpy array
				log_prob - the log probability of the selected action in the distribution
		"""

        temperature = max(1, self.initial_temperature)

        available_actions_idx = [self.env.system.get_transition_idx(tr_name) for tr_name in
                                 self.env.system.available_transitions()]

        # Query the actor network
        state = torch.tensor(state, dtype=torch.float).view(1, 1, -1)
        logits, actor_hidden_state = self.actor(state, actor_hidden_state)
        logits = logits.view(-1)
        filtered_logits = logits[available_actions_idx] / temperature

        dist = Categorical(logits=filtered_logits)

        # Sample an action from the distribution
        action = dist.sample()

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log probability of that action in our distribution
        return available_actions_idx[action.detach().numpy()], log_prob.detach(), actor_hidden_state, filtered_logits

    def evaluate(self, batch_obs, batch_acts):
        """
			Estimate the values of each observation, and the log probs of
			each action in the most recent batch with the most recent
			iteration of the actor network. Should be called from learn.
			Parameters:
				batch_obs - the observations from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of observation)
				batch_acts - the actions from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of action)
			Return:
				V - the predicted values of batch_obs
				log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
		"""

        batch_size = int(self.timesteps_per_batch / self.max_timesteps_per_episode)
        actor_hidden_state = self.actor.init_hidden(batch_size=batch_size)
        critic_hidden_state = self.critic.init_hidden(batch_size=batch_size)

        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V, critic_hidden_state = self.critic(batch_obs, critic_hidden_state)
        V = V.view(-1)

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        logits, actor_hidden_state = self.actor(batch_obs, actor_hidden_state)
        logits = logits.view(-1, self.tr_dim)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(batch_acts)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs

    def _init_hyperparameters(self, hyperparameters):
        """
			Initialize default and custom values for hyperparameters
			Parameters:
				hyperparameters - the extra arguments included when creating the PPO model, should only include
									hyperparameters defined below with custom values.
			Return:
				None
		"""
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.history_len = 1  # History length
        self.timesteps_per_batch = 500  # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 50  # Max number of timesteps per episode
        self.n_updates_per_iteration = 5  # Number of times to update actor/critic per iteration
        self.lr = 0.005  # Learning rate of actor optimizer
        self.gamma = 0.95  # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2  # Recommended 0.2, helps define the threshold to clip the ratio during SGA
        self.initial_temperature = 2
        self.temperature_decay = 1.008
        self.end_cond_threshold = 0.99

        # Miscellaneous parameters
        self.save_freq = 10  # How often we save in number of iterations
        self.seed = None  # Sets the seed of our program, used for reproducibility of results

        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

        # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert (type(self.seed) == int)

            # Set the seed
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def _log_summary(self):
        """
			Print to stdout what we've logged so far in the most recent batch.
			Parameters:
				None
			Return:
				None
		"""
        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []
