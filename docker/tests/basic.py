from __future__ import print_function
import numpy
import pandas
from athena.framework import Framework
from athena.model import AdditiveModel
from athena.helpers import split_dataframe
from athena.equations import *
from athena.dataset import Dataset


x = numpy.linspace(0.0, 1.0, 10000)
y = x + 5.0 + numpy.random.uniform(-0.1, 0.1, *x.shape)
df = pandas.DataFrame(data={"x": x, "y": y})

fw = Framework()
A, B = split_dataframe(df, 0.9)
fw.add_dataset(Dataset(A, B))

model = AdditiveModel(fw)
model.add(Bias)
model.add(MultiPolynomial, "x")
fw.initialize(model, A["y"].values)

fw.train()
print(fw.produce_equation())