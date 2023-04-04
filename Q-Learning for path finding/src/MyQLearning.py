from QLearning import QLearning
import numpy as np
class MyQLearning(QLearning):
    def update_q(self, state, action, r, state_next, possible_actions, alfa, gamma):
        # TODO Auto-generated method stub
        Q_old=self.get_q(state,action)
        if len(possible_actions)==0:
            Q_next_best=0
        else:
            Q_next_best=np.amax(self.get_action_values(state_next,possible_actions))
        Q_new=Q_old+alfa*(r+gamma*(Q_next_best-Q_old))
        self.set_q(state,action,Q_new)
        return
