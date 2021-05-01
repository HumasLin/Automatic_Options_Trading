import os
import pandas as pd
from datetime import date, datetime
from data_engineering import *
from train import *
from predict import *


def main(optionsymbol,capital_test):
	train_set, test_set = data_engineer(optionsymbol)
	#data_set = pd.read_csv('itm-prices.csv')

	#data_set['date'] = [datetime.strptime(date, '%Y-%m-%d') for date in data_set['date']]

	#train_set = data_set[data_set['date']<datetime.strptime('2019-01-01','%Y-%m-%d').date()]
	#test_set = data_set[data_set['date']>=datetime.strptime('2019-01-01','%Y-%m-%d').date()]

	training_data, test_data = list(train_set['adj_prices']), list(test_set['adj_prices'])

	capital = training_data[0] * 5

	if not capital_test:
		capital_test = test_data[0] * 5
	else:
		capital_test = int(capital_test)

	training_result = q_learning(training_data, capital, 5)
	while training_result == 0:
	    training_result = q_learning(training_data, capital, 5)
	predict_result = predict_trade(test_data, capital_test)

	os.remove("predict_model")
	os.remove("target_model")
	os.remove("predict_model_backup")
	os.remove("target_model_backup")
	return predict_result




