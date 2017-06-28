import numpy as np
from ch4.policy_iteration import  PolicyIteration
from ch4.value_iteration import ValueIteration

if __name__ == "__main__":
    row_num = 4
    col_num = 4
    states_num = row_num * col_num
    actions_num = 4
    states_terminal = [0, 15]
    # action_probs init
    action_probs = np.zeros((states_num, actions_num))
    for i in range(states_num):
        for j in range(actions_num):
            action_probs[i][j] = 1 / actions_num
    action_probs[0] = 0
    action_probs[15] = 0

    trans_probs = np.zeros((states_num, actions_num, states_num))
    rewards = np.zeros((states_num, actions_num, states_num))
    rewards.fill(-1)
    up = 0
    right = 1
    down = 2
    left = 3
    p = 1
    for i in range(row_num):
        for j in range(col_num):
            if j > 0:
                trans_probs[i * col_num + j][left][i * col_num + j - 1] = p
            else:
                trans_probs[i * col_num + j][left][i * col_num + j] = p

            if i > 0:
                trans_probs[i * col_num + j][up][(i - 1) * col_num + j] = p
            else:
                trans_probs[i * col_num + j][up][i * col_num + j] = p

            if j < col_num - 1:
                trans_probs[i * col_num + j][right][i * col_num + j + 1] = p
            else:
                trans_probs[i * col_num + j][right][i * col_num + j] = p

            if i < row_num - 1:
                trans_probs[i * col_num + j][down][(i + 1) * col_num + j] = p
            else:
                trans_probs[i * col_num + j][down][i * col_num + j] = p
    # 0
    trans_probs[0][:] = 0
    # 15
    trans_probs[15][:] = 0

    policy_iteration = PolicyIteration(trans_probs, rewards, action_probs, states_num, actions_num, 1, states_terminal)
    policy_iteration.policy_evaluation()
    policy_iteration.show_v(row_num, col_num)
    policy_iteration.show_policy(row_num, col_num)

    policy_iteration = PolicyIteration(trans_probs, rewards, action_probs, states_num, actions_num, 1, states_terminal)
    policy_iteration.policy_iteration()
    policy_iteration.show_v(row_num, col_num)
    policy_iteration.show_policy(row_num, col_num)

    value_iteration = ValueIteration(trans_probs, rewards, action_probs, states_num, actions_num, 1, states_terminal)
    value_iteration.value_iteration()
    value_iteration.show_v(row_num, col_num)
    value_iteration.show_policy(row_num, col_num)

