import numpy as np

'''
This code reproduces the figure 3.5
in book http://people.inf.elte.hu/lorincz/Files/RL_2006/SuttonBook.pdf

Given policy and MDP, we can compute value function by solving system
of linear equation.

The equation in the linear system is given by Bellman Equation
v(s) = sigma_a(pi(s|a)sigma_s'(prob(s'|s,a)[r(s',a,s)+gamma*v(s')))
'''

if __name__ == '__main__':
    row_num = 5
    col_num = 5
    total_num = row_num * col_num
    up = 0
    right = 1
    down = 2
    left = 3
    action_num = 4
    p = 1.0
    gamma = 0.9
    trans_probs = np.zeros((total_num, action_num, total_num))
    rewards =  np.zeros((total_num, action_num, total_num))
    action_probs = np.zeros((total_num, action_num))
    action_probs.fill(0.25)
    for i in range(row_num):
        for j in range(col_num):
            if j > 0:
                trans_probs[i * col_num + j][left][i * col_num + j - 1] = p
            else:
                trans_probs[i * col_num + j][left][i * col_num + j] = p
                rewards[i * col_num + j][left][i * col_num + j] = -1

            if i > 0:
                trans_probs[i * col_num + j][up][(i - 1) * col_num + j] = p
            else:
                trans_probs[i * col_num + j][up][i * col_num + j] = p
                rewards[i * col_num + j][up][i * col_num + j] = -1

            if j < col_num - 1:
                trans_probs[i * col_num + j][right][i * col_num + j + 1] = p
            else:
                trans_probs[i * col_num + j][right][i * col_num + j] = p
                rewards[i * col_num + j][right][i * col_num + j] = -1

            if i < row_num - 1:
                trans_probs[i * col_num + j][down][(i + 1) * col_num + j] = p
            else:
                trans_probs[i * col_num + j][down][i * col_num + j] = p
                rewards[i * col_num + j][down][i * col_num + j] = -1
            # special case for A
            if i == 0 and j == 1:
                trans_probs[i * col_num + j][up][i * col_num + j] = 0
                trans_probs[i * col_num + j][right][i * col_num + j + 1] = 0
                trans_probs[i * col_num + j][down][(i + 1) * col_num + j] = 0
                trans_probs[i * col_num + j][left][i * col_num + j - 1] = 0
                trans_probs[i * col_num + j][up][21] = 1 # 21 is the index of A'
                trans_probs[i * col_num + j][right][21] = 1
                trans_probs[i * col_num + j][down][21] = 1
                trans_probs[i * col_num + j][left][21] = 1
                rewards[i * col_num + j][up][21] = 10
                rewards[i * col_num + j][right][21] = 10
                rewards[i * col_num + j][down][21] = 10
                rewards[i * col_num + j][left][21] = 10

            # special case for B
            if i == 0 and j == 3:
                trans_probs[i * col_num + j][up][i * col_num + j] = 0
                trans_probs[i * col_num + j][right][i * col_num + j + 1] = 0
                trans_probs[i * col_num + j][down][(i + 1) * col_num + j] = 0
                trans_probs[i * col_num + j][left][i * col_num + j - 1] = 0
                trans_probs[i * col_num + j][up][13] = 1 # 13 is the index of B'
                trans_probs[i * col_num + j][right][13] = 1
                trans_probs[i * col_num + j][down][13] = 1
                trans_probs[i * col_num + j][left][13] = 1
                rewards[i * col_num + j][up][13] = 5
                rewards[i * col_num + j][right][13] = 5
                rewards[i * col_num + j][down][13] = 5
                rewards[i * col_num + j][left][13] = 5

    bs = np.zeros(total_num)
    for s in range(total_num):
        for a in range(action_num):
            s1 = 0.0
            for s_prime in range(total_num):
                s1 += trans_probs[s][a][s_prime] * rewards[s][a][s_prime]
            bs[s] += action_probs[s][a] * s1

    ws = np.zeros((total_num, total_num))
    s = 0
    for s in range(total_num):
        for a in range(action_num):
            for s_prime in range(total_num):
                ws[s][s_prime] += action_probs[s][a] * gamma * trans_probs[s][a][s_prime]
        ws[s] *= (-1) # move right side of equation to the left results in negative of the weight
        ws[s][s] += 1 # take into account the left side state weight
    # solving system of linear equations
    vs = np.linalg.solve(ws,bs)
    
    for row in range(row_num):
        for col in range(col_num):
            print("%.1f" % vs[row * col_num + col],end=' ')
        print()
