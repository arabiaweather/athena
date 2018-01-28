# Athena
[![Documentation Status](https://readthedocs.org/projects/arabiaweather-athena/badge/?version=latest)](http://arabiaweather-athena.readthedocs.io/en/latest/?badge=latest) [![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](http://www.gnu.org/licenses/lgpl-3.0)

## Accelerated equation building for academia and research
Athena is a high-level framework for equation building and curve fitting, written in Python and built on top of [Tensorflow](https://github.com/tensorflow/tensorflow); this means you can build large equations and perform curve fitting on your CPU, GPU, or cluster, without the constraints of traditional curve fitting toolboxes or any degradation in performance. Athena was developed with academia and researchers in mind: it is therefore abstract and simple to use (quickly fit an equation of choice to tabular data), while still remaining powerful and highly customizable (automatically search through millions of different mathematical equation forms and find the most accurate one).

## What can Athena do?

* Fit an arbitrary length mathematical equation to large amounts of data.
* Run equation building algorithms on a GPU or a cluster for increased performance.
* Automatically select the best features and choose the most suitable equation types.
* Search heuristically through different equation types (grid, random, genetic, etc).

## Getting started
The easiest way to install Athena and all its dependencies is through `pip`:

```
pip install git+git://github.com/arabiaweather/athena.git
```

## Building your first equation

Working with Athena can be as simple or as advanced as you need it to be. To demonstrate Athena's equation building capabilities, we'll fit a straight line to noisy data.
```python
x = numpy.linspace(0.0, 1.0, 10000)
y = x + 5.0 + numpy.random.uniform(-0.1, 0.1, *x.shape)
df = pandas.DataFrame(data={"x": x, "y": y})
```

Everything in Athena starts and ends with a `Framework`. Optimization hyper-parameters are defined inside it, and your data-set and model are attached to it.
```python
fw = Framework()
A, B = split_dataframe(df, 0.9)
fw.add_dataset(Dataset(A, B))
```

Here comes the fun part: Athena has built in hundreds of equation types that you can add, multiply, and composite together. We'll add the `FlexiblePower` and `Bias` functions to our model to form a straight line equation.
```python
model = AdditiveModel(fw)
model.add(Bias)
model.add(MultiPolynomial, "x")
fw.initialize(model, A["y"].values)
```

The only part left to get your equation is to train your model; this part can be sped up dramatically by using a CUDA-enabled GPU or by running Athena on a cluster. The result is very close to a straight line equation!
```python
fw.train()
print("y =", fw.produce_equation())
```

```
> y = 1.00098*x + 4.99821
```
The resulting equation can be pretty printed to a Python notebook, or better yet, can be converted to LaTeX for use in an academic paper easily.

## Diving into Athena

What makes any open source project great is the contributions of the community. Below are many great tutorials (in the form of Python notebooks) that show real world examples of powerful equation building and modelling techniques. You can contribute to this list too by submitting a pull request.

* [Modelling the Effect of Relative Humidity on Surface Temperature](https://github.com/arabiaweather/athena/blob/master/notebooks/temperature.ipynb)
* [Searching Through Randomly Generated Equations to Maximize Accuracy](https://github.com/arabiaweather/athena/blob/master/notebooks/searching.ipynb)
* [Using Athena's Built-in Genetic Algorithm to Search for and Interpret an Accurate Model](https://github.com/arabiaweather/athena/blob/master/notebooks/genetic.ipynb)
