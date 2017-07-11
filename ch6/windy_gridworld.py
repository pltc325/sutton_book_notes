from ch6.WindyAgent import WindyAgent


'''
Try to reproduce Figure 6.4
'''
if __name__ == "__main__":
    print("Trying to produce Figure 6.4")
    windy_agent = WindyAgent(70, 4, [30], [37], choose_action_strategy='epsilon_greedy', epsilon=0.1, alpha=0.5, gamma=1, episode_num=10000, row_num=7, col_num=10, step_max=100)
    windy_agent.run()
    for i in range(7):
        for j in range(10):
            print(windy_agent.pi[i*10+j],end=' ')
        print()
    windy_agent.run_greedily()