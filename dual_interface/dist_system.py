from process import Process, SUCCESS, FAIL, RANDOM


# a distributed system is a compound of processes (of class "Process").
class DistSystem(object):
    def __init__(self, name, processes):
        self.name = name
        self.processes = processes

    # reinitialize all processes in the system
    def reset(self):
        for pr in self.processes:
            pr.reset()

    # returns the process with that name
    def get_process(self, name):
        return next(proc for proc in self.processes if name == proc.name)

    # adds a new process to the system
    def add_process(self, process):
        self.processes.append(process)

    # adds a new transition to all the processes
    def add_transition(self, name, pr_list, source_list, target_list):
        for i in range(len(pr_list)):
            source = source_list[i]
            target = target_list[i]
            self.get_process(pr_list[i]).add_transition(name, source, target)


# This class inherits the properties of DistSystem. It is a specific case in which there are only two processes:
# a system and an environment. the system is the only process allowed to propose transitions to the environment.
# the system process has an RNN-based controller that can learn what are the optimal transitions to propose to the
# environment at any timestep.
class DualInterface(DistSystem):
    def __init__(self, name, if1: Process, if2: Process, history_len=1):
        super().__init__(name, [if1, if2])
        self.if1 = if1
        self.if2 = if2
        self.history_len = history_len
        self.if1_state_history = None
        self.if2_state_history = None
        self.if1_execution = []
        self.if2_execution = []

    def get_if(self, if_idx):
        if if_idx == 0:
            ret_obj = self.if1
        elif if_idx == 1:
            ret_obj = self.if2
        else:
            raise Exception("No such object")

        return ret_obj

    def reset(self):
        super().reset()
        self.init_state_history()
        self.init_executions()

    def init_state_history(self):
        self.if1_state_history = []
        self.if2_state_history = []
        for _ in range(self.history_len):
            if1_partial_state = self.if1.get_rnn_input(None)
            if2_partial_state = self.if2.get_rnn_input(None)
            self.if1_state_history.append(if1_partial_state)
            self.if2_state_history.append(if2_partial_state)

    def get_compound_state(self, if_idx):
        if if_idx == 0:
            compound_state_lst = self.if1_state_history[-self.history_len:]
        elif if_idx == 1:
            compound_state_lst = self.if2_state_history[-self.history_len:]
        else:
            compound_state_lst = []

        return [item for sublist in compound_state_lst for item in sublist]

    def init_executions(self):
        self.if1_execution = []
        self.if2_execution = []

    def step(self, if1_tr_idx, if2_tr_idx):
        success = False
        if1_tr = self.if1.transitions[if1_tr_idx].name
        if2_tr = self.if2.transitions[if2_tr_idx].name

        if if1_tr == if2_tr:
            full_if1_tr = self.if1.copy_transition_w_status(if1_tr, status=SUCCESS)
            full_if2_tr = self.if2.copy_transition_w_status(if2_tr, status=SUCCESS)

            self.if1_execution.append(self.if1.copy_transition_w_status(if1_tr, status=SUCCESS))
            self.if2_execution.append(self.if2.copy_transition_w_status(if2_tr, status=SUCCESS))

            # trigger the transition for both interfaces
            reward = max(self.if1.get_transition_reward(if1_tr), self.if2.get_transition_reward(if2_tr))
            self.if1.trigger_transition(if1_tr)
            self.if2.trigger_transition(if2_tr)
        else:
            full_if1_tr = self.if1.copy_transition_w_status(if1_tr, status=FAIL)
            full_if2_tr = self.if2.copy_transition_w_status(if2_tr, status=FAIL)

            self.if1_execution.append(self.if1.copy_transition_w_status(if1_tr, status=FAIL))
            self.if2_execution.append(self.if2.copy_transition_w_status(if2_tr, status=FAIL))

            if1_rnd_tr = self.if1.get_random_transition()
            if2_rnd_tr = self.if2.get_random_transition()

            # trigger the randomly chosen transitions for both interfaces
            reward = -1 * max(self.if1.get_transition_reward(if1_tr), self.if2.get_transition_reward(if2_tr))
            self.if1.trigger_transition(if1_rnd_tr)
            self.if2.trigger_transition(if2_rnd_tr)


        if1_partial_state = self.if1.get_rnn_input(full_if1_tr)
        if2_partial_state = self.if2.get_rnn_input(full_if2_tr)

        # building compound states for both interfaces
        self.if1_state_history.append(if1_partial_state)
        self.if2_state_history.append(if2_partial_state)

        if1_next_state = self.get_compound_state(0)
        if2_next_state = self.get_compound_state(1)

        return if1_next_state, if2_next_state, reward

    def local_step(self, if_tr_idx, if_idx):
        if if_idx == 0:
            if_tr = self.if1.transitions[if_tr_idx].name
            full_if_tr = self.if1.copy_transition_w_status(if_tr, status=SUCCESS)
            self.if1_execution.append(self.if1.copy_transition_w_status(if_tr, status=SUCCESS))

            # trigger the transition for the interface
            reward = self.if1.get_transition_reward(if_tr)
            self.if1.trigger_transition(if_tr)

            if_partial_state = self.if1.get_rnn_input(full_if_tr)

            # building compound states for both interfaces
            self.if1_state_history.append(if_partial_state)
            if_next_state = self.get_compound_state(0)

        elif if_idx == 1:
            if_tr = self.if2.transitions[if_tr_idx].name
            full_if_tr = self.if2.copy_transition_w_status(if_tr, status=SUCCESS)
            self.if2_execution.append(self.if2.copy_transition_w_status(if_tr, status=SUCCESS))

            # trigger the transition for the interface
            reward = self.if2.get_transition_reward(if_tr)
            self.if2.trigger_transition(if_tr)

            if_partial_state = self.if2.get_rnn_input(full_if_tr)

            # building compound states for both interfaces
            self.if2_state_history.append(if_partial_state)
            if_next_state = self.get_compound_state(1)

        else:
            raise Exception("No such interface")

        return if_next_state, reward

    def get_exploration_status(self, if_idx):
        return self.get_if(if_idx).get_exploration_status()

    def set_exploration_status(self, if_idx, status):
        self.get_if(if_idx).set_exploration_status(status)

    def invert_exploration_status(self, if_idx):
        status = self.get_exploration_status(if_idx)
        self.set_exploration_status(if_idx, not status)
