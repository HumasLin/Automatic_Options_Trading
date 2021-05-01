from util import *
from trader import *


def random_action(training_data, capital, window_size = 10):

    trader = Trader(window_size, True)
    num_steps = len(training_data) - 1

    state = state_normalize(training_data, 0, window_size + 1)
    total_profit = 0
    trader.portfolio = []
    random_profits = []

    for t in range(num_steps):

        action = np.random.randint(0, 3)

        # sit
        next_state = state_normalize(training_data, t + 1, window_size + 1)
        reward = 0
        # buy
        if action == 1:  
            if capital > training_data[t]:
                trader.portfolio.append(training_data[t])
                capital -= training_data[t]
        # sell
        elif action == 2:
            if len(trader.portfolio) > 0:
                bought_price = trader.portfolio.pop(0)
                reward = max(training_data[t] - bought_price, 0)
                total_profit += training_data[t] - bought_price
                capital += training_data[t]

        trader.history.push(state, action, next_state, reward)
        state = next_state

        random_profits.append(total_profit)

    return total_profit, random_profits
