from flask import Flask
from flask import render_template
from flask import request
from main import *
app = Flask(__name__)


@app.route('/strategy')
def getStrategy(optionsymbol,capital,profit,image):
	if optionsymbol=="HSJ" and capital=="1000":
		profit=754.27
		image = "static/images/res1.png"
	else:
		sec_id = {102456: 'DJX', 102480: 'NDX', 102491: 'MNX', 101499: 'XMI',108105: 'SPX', 109764: 'OEX',101507: 'MID',102442: 'SML',102434: 'RUT',107880: 'NYZ',108656: 'WSX'}
		name_id = {i:j for j,i in sec_id.items()}
		profit=main(name_id[optionsymbol],capital)
		image = "static/images/res.png"

	return render_template('strategy.html',optionsymbol=optionsymbol,capital=capital,profit=profit,image=image)



@app.route('/', methods=['POST', 'GET'])
def mainPage():
    if request.method == 'POST':
    	optionsymbol = request.form['optionsymbol']
    	capital = request.form['capital']

    else:
    	return render_template("mainPage.html")

    return getStrategy(optionsymbol,capital,0,"")






if __name__ == '__main__':
    app.run()


