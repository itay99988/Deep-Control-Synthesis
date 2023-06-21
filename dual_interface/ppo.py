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

    def __init__(self, policy_class, dual_if, **hyperparameters):
        """
			Initializes the PPO model, including hyperparameters.
			Parameters:
				policy_class - the policy class to use for our actor/critic networks.
				env - the environment to train on.
				hyperparameters - all extra arguments passed into PPO that should be hyperparameters.
			Returns:
				None
		"""
        # Initialize lists
        self.tr_dim, self.input_dim, self.actor, self.critic, self.actor_optim, self.critic_optim = [], [], [], [], [], []

        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(hyperparameters)

        # Extract dual interface environment information
        self.dual_if = dual_if
        self.tr_dim.append(len(dual_if.if1.transitions))
        self.tr_dim.append(len(dual_if.if2.transitions))
        self.input_dim.append(len(dual_if.if1.states) * self.tr_dim[0] * self.history_len)
        self.input_dim.append(len(dual_if.if2.states) * self.tr_dim[1] * self.history_len)
        self.end_condition = self.end_cond_threshold * self.max_timesteps_per_episode

        # Initialize actor and critic networks for both interfaces
        self.actor.append(policy_class(self.input_dim[0], self.tr_dim[0]))  # ALG STEP 1
        self.critic.append(policy_class(self.input_dim[0], 1))
        self.actor.append(policy_class(self.input_dim[1], self.tr_dim[1]))
        self.critic.append(policy_class(self.input_dim[1], 1))

        # Initialize optimizers for actor and critic
        self.actor_optim.append(Adam(self.actor[0].parameters(), lr=self.lr))
        self.critic_optim.append(Adam(self.critic[0].parameters(), lr=self.lr))
        self.actor_optim.append(Adam(self.actor[1].parameters(), lr=self.lr))
        self.critic_optim.append(Adam(self.critic[1].parameters(), lr=self.lr))

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
			Train the actor and critic networks. Here is where the main PPO algorithm resides.
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
        self.dual_if.set_exploration_status(0, True)
        self.dual_if.set_exploration_status(1, True)

        while t_so_far < total_timesteps:  # ALG STEP 2
            self.initial_temperature /= self.temperature_decay

            # Here we're collecting our batch simulations
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()  # ALG STEP 3

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens[0])

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # convert batch_obs to rnn batch, for each one of the interfaces
            batch_size = int(self.timesteps_per_batch / self.max_timesteps_per_episode)
            batch_obs[0] = batch_obs[0].view(batch_size, self.max_timesteps_per_episode, -1)
            batch_obs[1] = batch_obs[1].view(batch_size, self.max_timesteps_per_episode, -1)

            # Calculate advantage at k-th iteration
            V = [None, None]
            A_k = [None, None]
            V[0], _ = self.evaluate(batch_obs[0], batch_acts[0], 0)
            V[1], _ = self.evaluate(batch_obs[1], batch_acts[1], 1)
            A_k[0] = batch_rtgs[0] - V[0].detach()  # ALG STEP 5
            A_k[1] = batch_rtgs[1] - V[1].detach()

            # One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
            # isn't theoretically necessary, but in practice it decreases the variance of
            # our advantages and makes convergence much more stable and faster. I added this because
            # solving some environments was too unstable without it.
            A_k[0] = (A_k[0] - A_k[0].mean()) / (A_k[0].std() + 1e-10)
            A_k[1] = (A_k[1] - A_k[1].mean()) / (A_k[1].std() + 1e-10)

            # This is the loop where we update our network for some n epochs
            for _ in range(self.n_updates_per_iteration):  # ALG STEP 6 & 7
                curr_log_probs = [None, None]
                # Calculate V_phi and pi_theta(a_t | s_t)
                V[0], curr_log_probs[0] = self.evaluate(batch_obs[0], batch_acts[0], 0)
                V[1], curr_log_probs[1] = self.evaluate(batch_obs[1], batch_acts[1], 1)

                # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                # NOTE: we just subtract the logs, which is the same as
                # dividing the values and then canceling the log with e^log.
                # For why we use log probabilities instead of actual probabilities,
                # here's a great explanation:
                # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
                # TL;DR makes gradient ascent easier behind the scenes.
                ratios = [None, None]
                ratios[0] = torch.exp(curr_log_probs[0] - batch_log_probs[0])
                ratios[1] = torch.exp(curr_log_probs[1] - batch_log_probs[1])

                # Calculate surrogate losses.
                surr1, surr2 = [None, None], [None, None]
                surr1[0] = ratios[0] * A_k[0]
                surr1[1] = ratios[1] * A_k[1]
                surr2[0] = torch.clamp(ratios[0], 1 - self.clip, 1 + self.clip) * A_k[0]
                surr2[1] = torch.clamp(ratios[1], 1 - self.clip, 1 + self.clip) * A_k[1]

                # Calculate actor and critic losses.
                # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
                # the performance function, but Adam minimizes the loss. So minimizing the negative
                # performance function maximizes it.
                actor_loss, critic_loss = [None, None], [None, None]
                actor_loss[0] = (-torch.min(surr1[0], surr2[0])).mean()
                actor_loss[1] = (-torch.min(surr1[1], surr2[1])).mean()
                critic_loss[0] = nn.MSELoss()(V[0], batch_rtgs[0])
                critic_loss[1] = nn.MSELoss()(V[1], batch_rtgs[1])

                # Calculate gradients and perform backward propagation for actor and critic network
                self.actor_optim[0].zero_grad()
                actor_loss[0].backward(retain_graph=True)
                self.actor_optim[0].step()
                self.critic_optim[0].zero_grad()
                critic_loss[0].backward()
                self.critic_optim[0].step()

                self.actor_optim[1].zero_grad()
                actor_loss[1].backward(retain_graph=True)
                self.actor_optim[1].step()
                self.critic_optim[1].zero_grad()
                critic_loss[1].backward()
                self.critic_optim[1].step()

                # Log actor loss
                self.logger['actor_losses'].append(actor_loss[0].detach())

            # end condition
            last_ear = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews1']])
            avg_ep_return_lst.append(last_ear)
            avg_ep_return = np.mean(avg_ep_return_lst[-8:])
            if avg_ep_return >= self.end_condition:
                torch.save(self.actor[0].state_dict(), './ppo_actor1.pth')
                torch.save(self.actor[1].state_dict(), './ppo_actor2.pth')
                torch.save(self.critic[0].state_dict(), './ppo_critic1.pth')
                torch.save(self.critic[1].state_dict(), './ppo_critic2.pth')
                break

            # Print a summary of our training so far
            self._log_summary()

            # Save our model if it's time
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor[0].state_dict(), './ppo_actor1.pth')
                torch.save(self.actor[1].state_dict(), './ppo_actor2.pth')
                torch.save(self.critic[0].state_dict(), './ppo_critic1.pth')
                torch.save(self.critic[1].state_dict(), './ppo_critic2.pth')

    def rollout(self):
        """
			Parameters:
				None
			Return:
				batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
				batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
				batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
				batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
				batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
		"""

        # get exploration values for the interfaces
        exp_status = [self.dual_if.get_exploration_status(0), self.dual_if.get_exploration_status(1)]

        # states lists (for each interface)
        state = [[], []]
        next_state = [[], []]
        actor_hidden_state = [[], []]

        # Batch data. For more details, check function header.
        batch_obs = [[], []]
        batch_acts = [[], []]
        batch_log_probs = [[], []]
        batch_rews = [[], []]
        batch_rtgs = [[], []]
        batch_lens = [[], []]

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = [[], []]

        t = 0  # Keeps track of how many timesteps we've run so far this batch

        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < self.timesteps_per_batch:
            ep_rews[0] = []  # rewards collected per episode
            ep_rews[1] = []

            # Reset the environment. sNote that obs is short for observation.
            self.dual_if.reset()
            state[0] = self.dual_if.get_compound_state(0)
            state[1] = self.dual_if.get_compound_state(1)

            actor_hidden_state[0] = self.actor[0].init_hidden()
            actor_hidden_state[1] = self.actor[1].init_hidden()

            triggered_tr1 = triggered_tr2 = True
            actions_count1 = actions_count2 = 0

            t += self.max_timesteps_per_episode
            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            while max(actions_count1, actions_count2) < self.max_timesteps_per_episode:

                # t += 1  # Increment timesteps ran this batch so far

                # Calculate action and make a step, in both interfaces
                if triggered_tr1:
                    action1, log_prob1, actor_hidden_state[0], fil_logits1 = self.get_action(state[0],
                                                                                             actor_hidden_state[0],
                                                                                             if_idx=0,
                                                                                             exploration=exp_status[0])
                if triggered_tr2:
                    action2, log_prob2, actor_hidden_state[1], fil_logits2 = self.get_action(state[1],
                                                                                             actor_hidden_state[1],
                                                                                             if_idx=1,
                                                                                             exploration=exp_status[1])

                # both actions are global, or at least one is local?
                if self.dual_if.get_if(0).get_transition_by_idx(action1).is_global() and \
                   self.dual_if.get_if(1).get_transition_by_idx(action2).is_global():
                    next_state[0], next_state[1], reward1 = self.dual_if.step(action1, action2)
                    reward2 = reward1
                    triggered_tr1 = triggered_tr2 = True
                    actions_count1 += 1
                    actions_count2 += 1
                else:
                    if not self.dual_if.get_if(0).get_transition_by_idx(action1).is_global():
                        next_state[0], reward1 = self.dual_if.local_step(action1, 0)
                        triggered_tr1 = True
                        actions_count1 += 1
                    else:
                        triggered_tr1 = False

                    if not self.dual_if.get_if(1).get_transition_by_idx(action2).is_global():
                        next_state[1], reward2 = self.dual_if.local_step(action2, 1)
                        triggered_tr2 = True
                        actions_count2 += 1
                    else:
                        triggered_tr2 = False

                # Track recent observation, reward, action, and action log probability (if there was a progress)
                if triggered_tr1:
                    batch_obs[0].append(state[0])
                    ep_rews[0].append(reward1)
                    batch_acts[0].append(action1)
                    batch_log_probs[0].append(log_prob1)

                    state[0] = next_state[0]

                if triggered_tr2:
                    batch_obs[1].append(state[1])
                    ep_rews[1].append(reward2)
                    batch_acts[1].append(action2)
                    batch_log_probs[1].append(log_prob2)

                    state[1] = next_state[1]

                # if t == 1:
                #     print(F.softmax(fil_logits, dim=0))


            # Track episodic lengths and rewards
            batch_lens[0].append(self.max_timesteps_per_episode)
            batch_lens[1].append(self.max_timesteps_per_episode)

            # Fill the shorter episode with padding
            pad_length = self.max_timesteps_per_episode - min(actions_count1, actions_count2)
            if actions_count1 < actions_count2:
                for _ in range(pad_length):
                    batch_obs[0].append([0]*len(state[0]))
                    ep_rews[0].append(0)
                    batch_acts[0].append(0)
                    batch_log_probs[0].append(torch.tensor(0, dtype=torch.float))
            else:
                for _ in range(pad_length):
                    batch_obs[1].append([0]*len(state[1]))
                    ep_rews[1].append(0)
                    batch_acts[1].append(0)
                    batch_log_probs[1].append(torch.tensor(0, dtype=torch.float))

            # Do this anyway
            batch_rews[0].append(ep_rews[0])
            batch_rews[1].append(ep_rews[1])

        # print actions
        # print(self.get_actions_seq(batch_acts[0][0:self.max_timesteps_per_episode], 0))
        # print(self.get_actions_seq(batch_acts[1][0:self.max_timesteps_per_episode], 1))

        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs[0] = torch.tensor(batch_obs[0], dtype=torch.float)
        batch_obs[1] = torch.tensor(batch_obs[1], dtype=torch.float)
        batch_acts[0] = torch.tensor(batch_acts[0], dtype=torch.float)
        batch_acts[1] = torch.tensor(batch_acts[1], dtype=torch.float)
        batch_log_probs[0] = torch.tensor(batch_log_probs[0], dtype=torch.float)
        batch_log_probs[1] = torch.tensor(batch_log_probs[1], dtype=torch.float)
        batch_rtgs[0] = self.compute_rtgs(batch_rews[0])  # ALG STEP 4
        batch_rtgs[1] = self.compute_rtgs(batch_rews[1])

        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews1'] = batch_rews[0]
        self.logger['batch_rews2'] = batch_rews[1]
        self.logger['batch_lens'] = batch_lens[0]

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

    def get_action(self, state, actor_hidden_state, if_idx, exploration=True):
        """
			Queries an action from the actor network, should be called from rollout.
			Parameters:
				state - the observation at the current timestep
			Return:
				action - the action to take, as a numpy array
				log_prob - the log probability of the selected action in the distribution
		"""

        temperature = max(1, self.initial_temperature)

        if if_idx == 0:
            available_actions_idx = [self.dual_if.if1.get_transition_idx(tr_name) for tr_name in
                                     self.dual_if.if1.available_transitions()]
        elif if_idx == 1:
            available_actions_idx = [self.dual_if.if2.get_transition_idx(tr_name) for tr_name in
                                     self.dual_if.if2.available_transitions()]
        else:
            raise Exception("No such interface")


        # Query the actor network
        state = torch.tensor(state, dtype=torch.float).view(1, 1, -1)
        logits, actor_hidden_state = self.actor[if_idx](state, actor_hidden_state)
        logits = logits.view(-1)
        filtered_logits = logits[available_actions_idx] / temperature
        dist = Categorical(logits=filtered_logits)
        if exploration:
            # Sample an action from the distribution
            action = dist.sample()
        else:
            # Select the action with the highest probability
            action = torch.argmax(filtered_logits)

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log probability of that action in our distribution
        return available_actions_idx[action.detach().numpy()], log_prob.detach(), actor_hidden_state, filtered_logits

    def evaluate(self, batch_obs, batch_acts, if_idx):
        """
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
        actor_hidden_state = self.actor[if_idx].init_hidden(batch_size=batch_size)
        critic_hidden_state = self.critic[if_idx].init_hidden(batch_size=batch_size)

        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V, critic_hidden_state = self.critic[if_idx](batch_obs, critic_hidden_state)
        V = V.view(-1)

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        logits, actor_hidden_state = self.actor[if_idx](batch_obs, actor_hidden_state)
        logits = logits.view(-1, self.tr_dim[if_idx])
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(batch_acts)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs

    def get_actions_seq(self, act_lst, if_num):
        return [self.dual_if.get_if(if_num).get_transition_by_idx(act).name for act in act_lst]

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
        avg_ep_rews1 = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews1']])
        avg_ep_rews2 = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews2']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews1 = str(round(avg_ep_rews1, 2))
        avg_ep_rews2 = str(round(avg_ep_rews2, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Interface 1 Average Episodic Return: {avg_ep_rews1}", flush=True)
        print(f"Interface 2 Average Episodic Return: {avg_ep_rews2}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []
