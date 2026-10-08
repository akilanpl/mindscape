from dataclasses import asdict


class WorkingMemory:
    def reset(self, observation, state, goal):
        self.current_state = state
        self.current_goal = goal
        self.current_observation = asdict(observation)
        self.action_history = []
        self.event_history = []
        self.result_history = []
        self.trajectory_so_far = []

    def update(self, transition):
        self.current_state = transition.state_after
        self.action_history.append(transition.action)
        self.event_history.append(transition.event)
        self.result_history.append(transition.result)
        self.trajectory_so_far.append(transition)
        self.current_observation = {"kind": "observation", "source": "environment_feedback",
                                    "event": asdict(transition.event),
                                    "actual_result": asdict(transition.result)}
