from environment import State

class BaselineMethod:
    def __init__(self, wait_amount: int):
        self.wait_amount = wait_amount
    
    def predict(self, state: State):
        if state.time % self.wait_amount == 0:
            return 1
        else:
            return 0