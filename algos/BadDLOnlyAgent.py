from envs.SGAI_2048 import Act2048


class BadDLOnlyAgent:
    def __init__(self):
        self.last_action: Act2048 = Act2048.LEFT

    def get_action(self):
        if self.last_action == Act2048.LEFT:
            action = Act2048.DOWN
            self.last_action = Act2048.DOWN
        else:
            action = Act2048.LEFT
            self.last_action = Act2048.LEFT
        return action.value
