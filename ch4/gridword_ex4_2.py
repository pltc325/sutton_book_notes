import numpy as np
from policy_iteration import  PolicyIteration

if __name__ == "__main__":
    row_num = 5
    col_num = 4
    states_num = row_num * col_num
    actions_num = 4
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

            if i < row_num - 2:
                trans_probs[i * col_num + j][down][(i + 1) * col_num + j] = p
            else:
                trans_probs[i * col_num + j][down][i * col_num + j] = p
    # 0
    trans_probs[0][:] = 0
    # 15
    trans_probs[15][:] = 0
    # 16
    trans_probs[16][:] = 0
    # 18
    trans_probs[18][:] = 0
    # 19
    trans_probs[19][:] = 0
    trans_probs[17][up][13] = 1
    trans_probs[17][left][12] = 1
    trans_probs[17][left][16] = 0
    trans_probs[17][right][14] = 1
    trans_probs[17][right][18] = 0
    trans_probs[17][down][17] = 1

    trans_probs[13][down][17] = 1
    trans_probs[13][down][13] = 0

    policy_iteration = PolicyIteration(trans_probs, rewards, action_probs, 0.00001, states_num, actions_num, 1)
    policy_iteration.policy_evaluation()
    for i in range(row_num):
        for j in range(col_num):
            print("%.2f" % policy_iteration.v[i * col_num + j], end=" ")
        print()