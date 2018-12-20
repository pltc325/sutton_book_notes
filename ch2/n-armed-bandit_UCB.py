import numpy as np
import matplotlib.pyplot as plt
from utils import NArmedBandit
from utils import TestBed
'''
This code compares UCB(c=2) with eps-greedy(eps=0.1)

The result shows that:
1. UCB outperforms eps-greedy in terms of average reward during the phase where its optimal action is constantly smaller than that of eps-greedy.
This is because its suboptimal choice is better than the overall(average) performance of the eps-part choices made by eps-greedy.

I tried UCB(c=0.5), it got better result in terms of average reward as well as optimal action percentage.
'''


if __name__ == '__main__':
    num_steps = 3000
    num_arms = 10
    epsilons = [0.1, 0.01, 0]
    with_noise = False
    bandit_eps_greedy = NArmedBandit("eps-greedy", num_arms, num_steps, with_noise, {'eps-greedy': {'eps': 0.1}})
    bandit_UCB = NArmedBandit("UCB", num_arms, num_steps, with_noise, {'UCB': {'c': 0.5}})
    bandits = [bandit_eps_greedy, bandit_UCB]

    num_time = 500
    testbed = TestBed(bandits, num_time=num_time)

    testbed.run()
    testbed.show()
