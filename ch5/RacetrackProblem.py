import numpy as np
import random

'''
Exercise 5.4: Racetrack(no challenging version)

Use monte carole on policy method to teach a car to race to the terminal point as fast as possible while 
doesn't go off the track.

I made the following assumptions:
a) Once the car goes off the track, reset its position first one cell advanced vertically, if not possible,
one cell advanced horizontally
b) Once the car goes off the track, after being reset, its velocity is (0, 0)
c) It is said to be success only if the car sits right on the ending line after taking an action, this is
rarely the case in reality though.

It's also worth noting that if you don't reset the car position in case of being off the track, instead you
restart the race, the only way to make it learn is not to use accumulated reward but use the final reward,
which is either -5 or -1, to update the action value function, this is because it's probably more rewarded 
to just go off the track than to explore on the track.
'''


class Racetrack(object):
    def __init__(self,epsilon=0.1,action_num=9):
        self.action_num = action_num
        self.epsilon = epsilon
        self.track = np.zeros((32,17)).astype(int)
        # simulate the track
        self.track[:, 0] = self.build_col_of_track(4, 18)
        self.track[:, 1] = self.build_col_of_track(3, 10)
        self.track[:, 2] = self.build_col_of_track(1, 3)
        self.track[:, 3] = self.build_col_of_track(0, 0)
        self.track[:, 4] = self.build_col_of_track(0, 0)
        self.track[:, 5] = self.build_col_of_track(0, 0)
        self.track[:, 6] = self.build_col_of_track(0, 0)
        self.track[:, 7] = self.build_col_of_track(0, 0)
        self.track[:, 8] = self.build_col_of_track(0, 0)
        self.track[:, 9] = self.build_col_of_track(0, 25)
        self.track[:, 10] = self.build_col_of_track(0, 26)
        self.track[:, 11] = self.build_col_of_track(0, 26)
        self.track[:, 12] = self.build_col_of_track(0, 26)
        self.track[:, 13] = self.build_col_of_track(0, 26)
        self.track[:, 14] = self.build_col_of_track(0, 26)
        self.track[:, 15] = self.build_col_of_track(0, 26)
        self.track[:, 16] = self.build_col_of_track(0, 26)
        # we only use the first (31,3) in our code
        self.start_points = [(31, 3),(31, 4),(31, 5),(31, 6),(31, 7),(31, 8)]
        self.end_points = [(0, 16),(1, 16),(2, 16),(3, 16),(4, 16),(5, 16)]
        self.q = {}
        self.q_counts = {}
        self.q_avg = {}
        self.statistic = {'success': 0, 'fail': 2}
        # we start to count when the car at least succeeds once
        self.begin_count = False
        self.stop = False
        # initialize action value related functions
        for i in range(32):
            for j in range(17):
                if self.track[i][j]  == 1:
                    for v_vertical in range(-5,1):
                        for v_horizontal in range(6):
                            for a_vertical in range(-1,2):
                                for a_horizontal in range(-1,2):
                                    self.q[(i,j,v_vertical,v_horizontal,a_vertical,a_horizontal)] = 0
                                    self.q_counts[(i, j, v_vertical, v_horizontal, a_vertical, a_horizontal)] = 0
                                    self.q_avg[(i, j, v_vertical, v_horizontal, a_vertical, a_horizontal)] = 0

        print(self.q)
        # initialize action mappings
        self.pi = {}
        self.pi_mapping = {0:(-1,-1),1:(-1,0),2:(-1,1),3:(0,-1),4:(0,0),5:(0,1),6:(1,-1),7:(1,0),8:(1,1)}
        self.pi_mapping_r = {v: k for k, v in self.pi_mapping.items()}
        for i in range(32):
            for j in range(17):
                if self.track[i][j]  == 1:
                    for v_vertical in range(-5,1):
                        for v_horizontal in range(5):
                            self.pi[(i,j,v_vertical,v_horizontal)] = 2
        print(self.pi)

        self.cur_states = (self.start_points[0][0], self.start_points[0][1], 0, 0)

    def reset(self):
        """Reset the car to starting point
        :return: None
        """
        self.cur_states = (self.start_points[0][0], self.start_points[0][1], 0, 0)

    def step(self, action):
        """ Given action, change the car's position accordingly
        :param action: integer [0, 8]
        :return: None
        """
        a_v, a_h = self.pi_mapping[action]
        i, j, v_v, v_h = self.cur_states
        # since the car starts at bottom, the vertical velocity should be non 'positive'
        v_v_new, v_h_new = max(-4,min(0,v_v + a_v)), min(4,max(0,v_h + a_h))
        i_new, j_new = i + v_v_new, j + v_h_new
        ob = i_new, j_new, v_v_new, v_h_new
        # if it leads to the success
        if (i_new, j_new) in self.end_points:
            self.begin_count = True
            self.statistic['success'] += 1
            print("SUCCESS", i_new, j_new)
            # we want to stop the iteration once we have some confidence
            if self.statistic['success']/(self.statistic['fail'] + self.statistic['success'])>0.85:
                self.stop = True
            else:
                print(self.statistic['success']/(self.statistic['fail'] + self.statistic['success']))
            self.cur_states = ob
            # this may be trivial
            reward = -1
            done = True
        # if it leads to the accident
        elif self.is_off_track(i_new, j_new):
            if self.begin_count:
                self.statistic['fail'] += 1
            print("OFF TRACK, RESET POSITION AND CONTINUE", i_new, j_new)
            done = False
            # reset the car
            if not self.is_off_track(i-1,j):
                ob = i-1, j, 0, 0
                self.cur_states = ob
            elif not self.is_off_track(i, j+1):
                ob = i, j+1, 0, 0
                self.cur_states = ob
            else:
                raise ValueError("The car can go nowhere, it\'s weird.")
            # if after reset, the car succeeds
            if (ob[0], ob[1]) in self.end_points:
                done = True
            # penalize this action by -5
            reward = -5
        # if the car is on the way
        else:
            done = False
            self.cur_states = ob
            reward = -1
        return ob, reward, done

    def take_action(self, ob):
        """ tack action according to the observation using epsilon-soft policy
        :param ob: 
        :return: an integer, [0, 8]
        """
        p = np.random.rand()
        if p <= 1 - self.epsilon + self.epsilon / self.action_num:
            return self.pi[ob]
        else:
            a = self.pi[ob]
            candidates = list(range(self.action_num))
            candidates.remove(a)
            return random.choice(candidates)

    def update(self, obs_with_actions, rewards):
        """update the action value function and use it to find the new policy
        :param obs_with_actions: 
        :param reward: 
        :return: 
        """
        rewards_actual = []
        max_rewards = rewards[-1]
        rewards_actual.append(max_rewards)
        for i in range(len(rewards) - 1):
            rewards_actual.append(max_rewards - rewards[i])
        for i in range(len(obs_with_actions)):
            self.q[obs_with_actions[i]] += rewards_actual[i]
            self.q_counts[obs_with_actions[i]] += 1
            self.q_avg[obs_with_actions[i]] = self.q[obs_with_actions[i]] / self.q_counts[obs_with_actions[i]]

        for ob_with_action in obs_with_actions:
            ob = ob_with_action[0], ob_with_action[1], ob_with_action[2], ob_with_action[3]
            self.pi[ob] = self.find_max_action(ob_with_action)

        #print(self.q_avg[(28,6,-2,2,-1,1)])

    def find_max_action(self,ob_with_action):
        ob = ob_with_action[0], ob_with_action[1], ob_with_action[2], ob_with_action[3]
        if ob == (28, 6, -2, 2):
            a = 1
        max_q = -100000
        max_action = (0,0)
        for a_v in range(-1,2):
            for a_h in range(-1,2):
                ob_tmp = ob + (a_v, a_h)
                if ob_tmp in self.q_avg:
                    q = self.q_avg[ob_tmp]
                    if q > max_q:
                        max_q = self.q_avg[ob_tmp]
                        max_action = a_v, a_h
        return self.pi_mapping_r[max_action]

    def race(self, episode_num):
        done = False
        for episode in range(episode_num):
            print("episode:",episode)
            if self.stop:
                break
            self.reset()
            ob = self.cur_states
            this_suite_obs_with_action = []
            this_suite_rewards = []
            reward_sum = 0
            i = 0
            done = False
            while not done:
                #print("i",i)
                action = self.take_action(ob)
                #print("action",action)
                old_ob_with_action = ob + self.pi_mapping[action]
                this_suite_obs_with_action.append(old_ob_with_action)
                ob, reward, done = self.step(action)
                reward_sum += reward
                this_suite_rewards.append(reward_sum)
                i += 1
            self.update(this_suite_obs_with_action, this_suite_rewards)

    def show_track(self,track):
        for i in range(32):
            for j in range(17):
                if track[i][j] == 0:
                    print(end="  ")
                else:
                    print(track[i][j], end=" ")
            print()

    def build_col_of_track(self,up,down):
        col = np.ones(32)
        if up != 0:
            col[0:up] = 0
        if down != 0:
            col[-down:] = 0
        return col

    def is_off_track(self, i, j):
        # border check
        if i < 0 or i > 31:
            return True
        if j < 0 or j > 16:
            return True
        if self.track[i][j] == 0:
            return True
        return False

    def show_trajectory(self):
        self.reset()
        track_copy = self.track.copy()
        track_copy[(self.cur_states[0],self.cur_states[1])] = 6
        done = False
        while not done:
            action = self.pi[self.cur_states]
            print("action",action,self.pi_mapping[action])
            ob, reward, done = self.step(action)
            track_copy[(self.cur_states[0], self.cur_states[1])] = 6
        self.show_track(track_copy)


if __name__ == "__main__":
    racetrack = Racetrack()
    racetrack.show_track(racetrack.track)
    racetrack.race(2000000)
    racetrack.show_trajectory()