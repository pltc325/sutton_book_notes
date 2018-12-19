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
    bandits = [NArmedBandit(num_arms, epsilon, num_steps, with_noise=False, Q=None, alpha=alpha, is_stationary=False) for alpha in alphas]
    num_time = 100
    testbed = TestBed(bandits, num_time=num_time)

    testbed.run()
    testbed.show()