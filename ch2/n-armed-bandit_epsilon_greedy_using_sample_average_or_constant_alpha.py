from utils import NArmedBandit, TestBed

'''
This code compares sample average and 
constant alpha update rules using epsilon-greedy strategy
under a non-stationary situation.

We can see that constant alpha outperforms sample average, 
it's not a surprise as constant alpha learns faster when facing non-stationary environment.
'''

if __name__ == '__main__':
    num_times = 20
    num_steps = 200
    num_arms = 10
    epsilon = 0.1
    alphas = [None, 0.9]
    num_choices = len(alphas)
    bandit_sample_average = NArmedBandit("sample-average", num_arms, num_steps, with_noise=False,action_selection_strategy={'eps-greedy':{'eps':0.1}},
                                         Q_update_rule={'sample-average': None}, Q=None, is_stationary=False)
    bandit_constant_alpha = NArmedBandit("constant-alpha", num_arms, num_steps, with_noise=False,action_selection_strategy={'eps-greedy':{'eps':0.1}},
                                         Q_update_rule={'constant-alpha': 0.9}, Q=None, is_stationary=False)

    bandits = [bandit_sample_average, bandit_constant_alpha]
    num_time = 100
    testbed = TestBed(bandits, num_time=num_time)

    testbed.run()
    testbed.show()
