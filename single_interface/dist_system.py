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
class BlackboxEnvironment(DistSystem):
    def __init__(self, name, system: Process, environment: Process, history_len=1):
        super().__init__(name, [system, environment])
        self.system = system
        self.environment = environment
        self.history_len = history_len
        self.state_history = None
        self.system_execution = []
        self.environment_execution = []

    def reset(self):
        super().reset()
        self.init_state_history()
        self.init_executions()

    def init_state_history(self):
        self.state_history = []
        for _ in range(self.history_len):
            partial_state = self.system.get_rnn_input(None)
            self.state_history.append(partial_state)

    def get_compound_state(self):
        compound_state_lst = self.state_history[-self.history_len:]
        return [item for sublist in compound_state_lst for item in sublist]

    def init_executions(self):
        self.system_execution = []
        self.environment_execution = []

    def step(self, sys_tr_idx):
        success = False
        partial_state = None
        sys_tr = self.system.transitions[sys_tr_idx].name

        if self.environment.is_transition_enabled(sys_tr):
            full_sys_tr = self.system.copy_transition_w_status(sys_tr, status=SUCCESS)

            self.system_execution.append(self.system.copy_transition_w_status(sys_tr, status=SUCCESS))
            self.environment_execution.append(self.environment.copy_transition_w_status(sys_tr, status=SUCCESS))

            # trigger the transition for both system and environment
            self.system.trigger_transition(sys_tr)
            self.environment.trigger_transition(sys_tr)
            reward = 1
        else:
            full_sys_tr = self.system.copy_transition_w_status(sys_tr, status=FAIL)
            env_rnd_tr = self.environment.get_random_transition()

            self.system_execution.append(self.system.copy_transition_w_status(sys_tr, status=FAIL))
            self.environment_execution.append(self.environment.copy_transition_w_status(env_rnd_tr, status=RANDOM))

            # trigger the randomly chosen transition of the environment
            self.environment.trigger_transition(env_rnd_tr)
            reward = -1

        partial_state = self.system.get_rnn_input(full_sys_tr)

        # building compound state
        self.state_history.append(partial_state)
        next_state = self.get_compound_state()
        return next_state, reward
