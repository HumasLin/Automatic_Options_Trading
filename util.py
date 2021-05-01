import numpy as np
import matplotlib.pyplot as plt

sec_id = {102456: 'DJX',
          102480: 'NDX',
          102491: 'MNX',
          101499: 'XMI',
          108105: 'SPX',
          109764: 'OEX',
          101507: 'MID',
          102442: 'SML',
          102434: 'RUT',
          107880: 'NYZ',
          108656: 'WSX'}
          
# prints price in formatted form
def format_price(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# returns an an n-day state representation ending at time t, i.e. state in [t-n+1,t]
def state_normalize(data, t, n):
    d = t - n + 1
    if d >= 0:
        block = data[d:t + 1]
    else:
        block = -d * [data[0]] + data[0:t + 1] # pad with t0
    res = []

    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))

    return np.array([res])

def plot_comparison(model, random):
    fig, ax = plt.subplots(1, 1, figsize=(10,7), sharex=True, sharey=True, dpi=300)
    plt.plot(model)
    plt.plot(random)
    plt.xlabel("Time steps",fontsize=16)
    plt.xticks(size=16)
    plt.ylabel("Cumulative Reward",fontsize=16)
    plt.yticks(size=16)
    plt.legend(['Model', 'Random'], loc='best',fontsize=16)
    plt.title("Cumulative reward in option trading",fontsize=16)
    plt.savefig('model_random_comp.jpg')
    plt.close()

def plot_actions(closes, buys, sells, total_profit):
    x_data = range(len(closes))
    fig, ax = plt.subplots(1, 1, figsize=(10,7), sharex=True, sharey=True, dpi=300)
    plt.plot(x_data, closes,color='green')
    plt.plot(x_data, buys, marker='o', markersize=8, markerfacecolor='blue')
    plt.plot(x_data, sells, marker='o', markersize=8, markerfacecolor='orange')
    plt.xlabel("Time steps",fontsize=16)
    plt.xticks(size=16)
    plt.ylabel("Option Price",fontsize=16)
    plt.yticks(size=16)
    plt.legend(['Close Price', 'Buy', 'Sell'], loc='best',fontsize=16)
    plt.title("Trading actions on options with profit = {}".format(total_profit),fontsize=16)
    plt.savefig("static/images/res.png")
    plt.close()
