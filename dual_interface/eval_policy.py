import torch


def _log_summary(ep_len, ep_ret1, ep_ret2, ep_num):
    """
        Print to stdout what we've logged so far in the most recent episode.
        Parameters:
            None
        Return:
            None
    """
    # Round decimal places for more aesthetic logging messages
    failure_rate1 = str(round((ep_len - ep_ret1) * (50 / ep_len), 2))
    failure_rate2 = str(round((ep_len - ep_ret2) * (50 / ep_len), 2))
    ep_len = str(round(ep_len, 2))
    ep_ret1 = str(round(ep_ret1, 2))
    ep_ret2 = str(round(ep_ret2, 2))

    # Print logging statements
    print(flush=True)
    print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
    print(f"Episodic Length: {ep_len}", flush=True)
    print(f"Interface 1 Episodic Return: {ep_ret1}", flush=True)
    print(f"Interface 2 Episodic Return: {ep_ret2}", flush=True)
    print("Interface 1 Failure Rate: {}%".format(failure_rate1))
    print("Interface 2 Failure Rate: {}%".format(failure_rate2))
    print(f"------------------------------------------------------", flush=True)
    print(flush=True)


def rollout(policy1, policy2, dual_if):
    """
        Returns a generator to roll out each episode given a trained policy and
        environment to test on.
        Parameters:
            policy - The trained policy to test
            env - The environment to evaluate the policy on

        Return:
            A generator object rollout, or iterable, which will return the latest
            episodic length and return on each iteration of the generator.
    """
    # Rollout until user kills process
    for _ in range(10):
        dual_if.reset()
        state1 = dual_if.get_compound_state(0)
        state2 = dual_if.get_compound_state(1)
        hidden_state1 = policy1.init_hidden()
        hidden_state2 = policy2.init_hidden()

        # number of timesteps so far
        t = 0

        # Logging data
        ep_len = 0  # episodic length
        ep_ret1 = 0  # episodic return of interface1
        ep_ret2 = 0  # episodic return of interface2
        actions_count1 = 0
        actions_count2 = 0

        while max(actions_count1, actions_count2) < 200:
            t += 1

            # Query deterministic action from policy and run it
            state1 = torch.tensor(state1, dtype=torch.float).view(1, 1, -1)
            state2 = torch.tensor(state2, dtype=torch.float).view(1, 1, -1)

            # select the best action
            available_actions_idx1 = [dual_if.if1.get_transition_idx(tr_name) for tr_name in
                                      dual_if.if1.available_transitions()]
            available_actions_idx2 = [dual_if.if2.get_transition_idx(tr_name) for tr_name in
                                      dual_if.if2.available_transitions()]

            logits1, hidden_state1 = policy1(state1, hidden_state1)
            logits1 = logits1.view(-1)
            logits2, hidden_state2 = policy2(state2, hidden_state2)
            logits2 = logits2.view(-1)

            filtered_logits1 = logits1[available_actions_idx1]
            filtered_logits2 = logits2[available_actions_idx2]

            # choose based on probability
            # dist = Categorical(logits=filtered_logits)
            # action_idx = dist.sample()

            # choose action with maximal value
            action_idx1 = torch.argmax(filtered_logits1)
            action_idx2 = torch.argmax(filtered_logits2)

            real_action_idx1 = available_actions_idx1[action_idx1.numpy()]
            real_action_idx2 = available_actions_idx2[action_idx2.numpy()]

            # both actions are global, or at least one is local?
            if dual_if.get_if(0).get_transition_by_idx(real_action_idx1).is_global() and \
                    dual_if.get_if(1).get_transition_by_idx(real_action_idx2).is_global():
                state1, state2, reward1 = dual_if.step(real_action_idx1, real_action_idx2)
                reward2 = reward1
                actions_count1 += 1
                actions_count2 += 1
            else:
                if not dual_if.get_if(0).get_transition_by_idx(real_action_idx1).is_global():
                    state1, reward1 = dual_if.local_step(real_action_idx1, 0)
                    actions_count1 += 1

                if not dual_if.get_if(1).get_transition_by_idx(real_action_idx2).is_global():
                    state2, reward2 = dual_if.local_step(real_action_idx2, 1)
                    actions_count2 += 1

            # Sum all episodic rewards as we go along
            ep_ret1 += reward1
            ep_ret2 += reward2

            # Reset reward values
            reward1 = reward2 = 0

        # Track episodic length
        ep_len = t

        # returns episodic length and return in this iteration
        yield ep_len, ep_ret1, ep_ret2


def eval_policy(policy1, policy2, dual_if):
    """
        Parameters:
            policy - The trained policy to test, basically another name for our actor model
            env - The environment to test the policy on
        Return:
            None
        NOTE: To learn more about generators, look at rollout's function description
    """
    # Rollout with the policy and environment, and log each episode's data
    for ep_num, (ep_len, ep_ret1, ep_ret2) in enumerate(rollout(policy1, policy2, dual_if)):
        _log_summary(ep_len=ep_len, ep_ret1=ep_ret1, ep_ret2=ep_ret2, ep_num=ep_num)
