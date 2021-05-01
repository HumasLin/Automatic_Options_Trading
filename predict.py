import torch
import numpy as np
from trader import Trader
from util import *

# test whether a model can help a trader make money
def predict_trade(test_data, capital, window_size = 10):

    total_profit = 0
    closes = []
    buys = []
    sells = []
    done = True
    trader = Trader(window_size, True)
    trader.portfolio = []
    num_steps = len(test_data) - 1
    batch_size = 32

    state = state_normalize(test_data, 0, window_size + 1)

    for t in range(num_steps):

        action = trader.act(state)
        closes.append(test_data[t])
        next_state = state_normalize(test_data, t + 1, window_size + 1)
        reward = 0
        # buy
        if action == 1: 
            if capital > test_data[t]:
                trader.portfolio.append(test_data[t])
                buys.append(test_data[t])
                sells.append(None)
                capital -= test_data[t]
            else:
                buys.append(None)
                sells.append(None)
        # sell
        elif action == 2: 
            if len(trader.portfolio) > 0:
                bought_price = trader.portfolio.pop(0)
                reward = max(test_data[t] - bought_price, 0)
                total_profit += test_data[t] - bought_price
                buys.append(None)
                sells.append(test_data[t])
                capital += test_data[t]
            else:
                buys.append(None)
                sells.append(None)
        elif action == 0:
            buys.append(None)
            sells.append(None)

        if t == num_steps - 1:
            done = True 

        trader.history.push(state, action, next_state, reward)
        state = next_state

    plot_actions(closes, buys, sells, total_profit)


    return total_profit
