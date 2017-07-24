from ch6.WindyAgent import WindyAgent


'''
Try to reproduce Figure 6.4 and do the exercises 6.7
'''
if __name__ == "__main__":
    print("Trying to produce Figure 6.4")
    # 4 actions: 'left', 'up', 'right', 'down'
    windy_agent = WindyAgent(state_num=70, action_num=4, start_states=[30],
                             terminal_states=[37], choose_action_strategy='epsilon_greedy',
                             epsilon=0.1, alpha=0.5, gamma=1, episode_num=10000,
                             row_num=7, col_num=10, step_max=100)
    windy_agent.run()
    windy_agent.run_greedily()
    windy_agent.print_path()

    # 8 actions: additional 'left-up', 'left-down', 'right-up', 'right-down'
    windy_agent = WindyAgent(state_num=70, action_num=8, start_states=[30],
                             terminal_states=[37], choose_action_strategy='epsilon_greedy',
                             epsilon=0.1, alpha=0.5, gamma=1, episode_num=10000,
                             row_num=7, col_num=10, step_max=100)
    windy_agent.run()
    windy_agent.run_greedily()
    windy_agent.print_path()

    # 9 actions: additional 'stay put' included
    windy_agent = WindyAgent(state_num=70, action_num=9, start_states=[30],
                             terminal_states=[37], choose_action_strategy='epsilon_greedy',
                             epsilon=0.1, alpha=0.5, gamma=1, episode_num=10000,
                             row_num=7, col_num=10, step_max=100)
    windy_agent.run()
    windy_agent.run_greedily()
    windy_agent.print_path()
