from util import *
from trader import Trader
from random_action import *
import time
import torch
import matplotlib.pyplot as plt

def q_learning(training_data, capital, num_episode=100, window_size=10):

    profits = []
    for i in range(10):
        profit, series = random_action(training_data, capital)
        profits.append(profit)
    mean_profit = np.mean(profits)
    while profit > mean_profit:
        profit, series = random_action(training_data, capital)

    trader = Trader(window_size)
    num_step = len(training_data) - 1

    starttime = time.time()
    for e in range(num_episode + 1):
        
        cumulative_reward=[]
        done = False
        state = state_normalize(training_data, 0, window_size + 1)

        total_profit = 0
        trader.portfolio = []
        x_data = range(num_step)

        for t in range(num_step):

            action = trader.act(state)

            next_state = state_normalize(training_data, t + 1, window_size + 1)
            reward = 0            
            # buy action
            if action == 1:
                if capital > training_data[t]:
                    trader.portfolio.append(training_data[t])
                    capital -= training_data[t]
            # sell action
            elif action == 2:
                if len(trader.portfolio) > 0:
                    bought_price = trader.portfolio.pop(0)
                    reward = max(training_data[t] - bought_price, 0)
                    total_profit += training_data[t] - bought_price
                    capital += training_data[t]
            if t == num_step - 1:
                done = True  

            trader.history.push(state, action, next_state, reward)
            state = next_state
            cumulative_reward.append(total_profit)
            trader.optimize()

        if total_profit==0:
            try:
                trader.predict_net = torch.load('predict_model_backup', map_location=device)
                trader.target_net = torch.load('target_model_backup', map_location=device)
            except:
                return 0
        else:
            torch.save(trader.predict_net, "predict_model_backup")
            torch.save(trader.target_net, "target_model_backup")

        if e % 10 == 0:
            trader.target_net.load_state_dict(trader.predict_net.state_dict())
            torch.save(trader.predict_net, "predict_model")
            torch.save(trader.target_net, "target_model")

    plot_comparison(cumulative_reward,series)
