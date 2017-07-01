import warnings
warnings.filterwarnings("ignore")

from os import environ
#environ["SYMPY_USE_CACHE"] = "no"
#print(environ["SYMPY_USE_CACHE"])

from pandas import read_csv
from athena.helpers import *
from athena.framework import Framework
from athena.dataset import Dataset
from athena.searching import GeneticSearch
from sympy import init_printing, N, nsimplify, Symbol, diff

experiment_results = []

df = read_csv('/home/khaled/repositories/athena-github/ivzhe139.csv', index_col=None)
df = df[['time', 'UV', 'baromin', 'humidity', 'light', 'rainin', 'temp', 'windgust']]

from dateutil.parser import parse


def get_hour (x):
	y = parse(x)
	return y.hour + y.minute / 60.0


df["time"] = [get_hour(x) / 24.0 for x in df["time"].values]

parameters_map = {
	"normalized"    : {
		"humidity": "rh",
		"windgust": "ws",
		"UV"      : "uv",
		"light"   : "light",
	},
	"not_normalized": {
		"time"  : "time",
		"rainin": "rain",
	},
	"target"        : "temp",
}

inverse_parameters_map = {}

for l in ["normalized", "not_normalized"]:
	for i, j in parameters_map[l].items():
		inverse_parameters_map[j] = i

framework_parameters = {
	"starting_lr"   : 0.1,
	"max_iterations": int(1e2),
	"momentum"      : 0.99,
}

fw = Framework(framework_parameters)

init_printing()

training_df, testing_df = split_dataframe(df, 0.9)
fw.add_dataset(Dataset(training_df, testing_df, parameters_map))
target_values = df[fw.dataset.target_column_name].values

reconstruction_list = None

search_length = 50
equation_length = 4
from sys import stdout

constituents_graph = {"Training Correlation" : {"x": [], "y": []},
                      "Testing Correlation"  : {"x": [], "y": []}}

from sympy import pprint, init_printing
init_printing()

gs = GeneticSearch(fw, search_length=search_length, equation_length=equation_length)
for iteration in range(100):
	print("=" * 40, iteration + 1, "=" * 40)

	equation = gs.iteration()

	constituents_graph["Training Correlation"]["x"].append(iteration)
	constituents_graph["Training Correlation"]["y"].append(equation["training_pearson"])

	constituents_graph["Testing Correlation"]["x"].append(iteration)
	constituents_graph["Testing Correlation"]["y"].append(equation["testing_pearson"])

	stdout.flush()

	for i, j in enumerate(sorted(equation["constituents"], key=lambda x: x["contribution"])[::-1]):
		c = j["contribution"]
		a = list(equation["reconstruction"][i]["args"])
		string = []
		for _a in a:
			if isinstance(_a, str):
				string.append(_a)
			else:
				string.append(str(_a.__name__))
		string = ' '.join(string)
		if string in constituents_graph:
			constituents_graph[string]["x"].append(iteration)
			constituents_graph[string]["y"].append(c)
		else:
			constituents_graph[string] = {"x": [iteration], "y": [c]}

	print("Training Correlation: {}%, Testing Correlation: {}%".format(equation["training_pearson"]*100, equation["testing_pearson"]*100))
	e = sum(equation["equation"])

	variables = e.atoms(Symbol)

	for v in variables:
		dt_dv = diff(e, v)
		dt_dv = N(dt_dv, 1)
		print("For the variable {0}, dT/d({0}) is simplified to:".format(v))
		print(dt_dv)
		print("\n")

import matplotlib.pyplot as plt

plt.figure()
for i, j in constituents_graph.items():
	if "Correlation" in i: continue
	plt.plot(j["x"], j["y"], label=i)

plt.legend()
plt.show()