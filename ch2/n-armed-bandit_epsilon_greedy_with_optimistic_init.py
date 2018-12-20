import numpy as np

from utils import NArmedBandit, TestBed

'''
Q=5, eps=0  > Q=1, eps=0.1
After warm-up phase, we're almost sure to find the best action, eps=0 makes us stick with it

Note we have to use 'constant-alpha'(not sample-average) here since it 'has' bias on the initial estimations of Q, this is desired when we 
set do optimistic initialization. 
'''

if __name__ == '__main__':
    num_steps = 100
    num_arms = 10
    alpha = None
    with_noise = True
    Q = np.ones((num_arms,)) * 5
    bandit_a = NArmedBandit("without optimistic init", num_arms, num_steps, with_noise, {'eps-greedy':{'eps':0.1}}, {'constant-alpha': 0.1}, Q=None)
    bandit_b = NArmedBandit("with optimistic init", num_arms, num_steps, with_noise, {'eps-greedy':{'eps':0}}, {'constant-alpha': 0.1}, Q=Q)
    bandits = [bandit_a, bandit_b]
    num_time = 500
    testbed = TestBed(bandits, num_time=num_time)

    testbed.run()
    testbed.show()
