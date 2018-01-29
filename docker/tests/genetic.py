from pandas import read_csv, DataFrame
import numpy as np
from athena.helpers import *
from athena.framework import Framework
from athena.dataset import Dataset
from athena.searching import RandomSearch
from sympy import init_printing, Symbol, lambdify, N

df = read_csv('/code/tests/weather-station.csv', index_col=None)
df = df[['time', 'UV', 'humidity', 'light', 'rainin', 'temp', 'windgust']]

from dateutil.parser import parse

def get_hour (x):
    y = parse(x)
    return y.hour + y.minute / 60.0

df["time"] = [get_hour(x) / 24.0 for x in df["time"].values]

parameters_map = {
    "normalized"  : 
    {
        "humidity": "rh",
        "windgust": "ws",
        "UV"      : "uv",
        "light"   : "light",
    },
    "not_normalized": 
    {
        "time"    : "time",
        "rainin"  : "rain",
    },
    "target"      : "temp",
}

inverse_parameters_map = {}

for l in ["normalized", "not_normalized"]:
    for i, j in parameters_map[l].items():
        inverse_parameters_map[j] = i

framework_parameters = {
    "starting_lr"   : 0.01,
    "max_iterations": int(1e4),
    "momentum"      : 0.99,
}

fw = Framework(framework_parameters)

init_printing()

training_df, testing_df = split_dataframe(df, 0.9)
fw.add_dataset(Dataset(training_df, testing_df, parameters_map))
target_values = df[fw.dataset.target_column_name].values

reconstruction_list = None
search_length = 100
equation_length = 1

for iteration in range(10):
    rs = RandomSearch(fw, search_length=search_length, 
                      equation_length=equation_length,
                      starting_equations=reconstruction_list)

    print("=" * 40, iteration + 1, "=" * 40)
    rs.search(return_constituents=True)
    equation = rs.get_best_equations(k=1)[0]
    
    print("\n" * 2)
    
    print("Correlation:", str(int(round(equation["testing_pearson"] * 100)))+"%")

    biases, equations = 0, []
    for i, cnst in enumerate(equation["equation"]):
        if len(cnst.atoms(Symbol)) == 0:
            biases += cnst
        else:
            equations += [cnst]

    equation["equation"] = [biases] + equations

    print("\n" * 2)
    print(sum(equation["equation"]))
    print("\n" * 2)

    contributions = []
    interval_ranges = []

    eq_df = []

    for i, cnst in enumerate(equation["equation"]):
        variables = cnst.atoms(Symbol)

        substitutions = [df[inverse_parameters_map[str(v)]].values for v in variables]
        function = lambdify(tuple(variables), cnst, "numpy")
        contribution = mean_confidence_interval(np.abs(function(*tuple(substitutions))) / target_values * 100.0, 0.99)

        eq_df.append({"Constituent": str(N(cnst,1)),
                      "Contribution": int(round(contribution[0]))})

    eq_df = DataFrame(data=eq_df)
    print(eq_df)


    reconstruction_list = equation["reconstruction"]









