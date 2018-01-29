import tensorflow as tf
import numpy as np
from sympy import symbols, sin, sech, pi, exp, log, Expr
from copy import copy


class Equation:
    def __init__(self, tf_graph):
        """
        Selection constructor function

        Athena uses a random forest from Scikit-Learn to select the best parameters. Use this constructor to create a Selection class.

        Parameters
        ----------
        tf_graph : ?
            Number of estimators (decision trees) the Random Forest will use.
        """
        self.tf_graph = tf_graph

        # TODO: the number of Tensorflow variables should not be fixed but instead should change dynamically as the model grows
        with self.tf_graph.as_default():
            self.w = tf.Variable(tf.random_normal([1000], mean=0, stddev=1), name='coefficients')

        self._wc = int(0)
        self.equations = []
        self.datasets = {}

    def produce_equations(self, session):
        """
        Selection constructor function

        Athena uses a random forest from Scikit-Learn to select the best parameters. Use this constructor to create a Selection class.

        Parameters
        ----------
        session : ?
            Number of estimators (decision trees) the Random Forest will use.
        """
        from tensorflow.python.framework.ops import Tensor
        from sympy.core.symbol import Symbol

        equation = []
        self_equations = []
        for e in self.equations:
            self_equations.append(copy(e))

        for e in self_equations:
            for sub in e["substitutions"]:
                assert isinstance(sub[1], Symbol)

                # TODO: investigate why the following line produces a KeyError in Sympy
                e["equation-string"] = e["equation-string"].subs(sub[1], session.run(sub[0]) if isinstance(sub[0], Tensor) else sub[0])

            equation.append(e["equation-string"])

        return equation

    def set_weights(self, new_weights):
        """
        Selection constructor function

        Athena uses a random forest from Scikit-Learn to select the best parameters. Use this constructor to create a Selection class.

        Parameters
        ----------
        new_weights : ?
            Number of estimators (decision trees) the Random Forest will use.
        """
        return self.w.assign(new_weights)

    # TODO: change min and max names so they don't shadow the built in functions
    @staticmethod
    def normalize(_array, _string, min=None, max=None):
        """
        Selection constructor function

        Athena uses a random forest from Scikit-Learn to select the best parameters. Use this constructor to create a Selection class.

        Parameters
        ----------
        _array : list
            Number of estimators (decision trees) the Random Forest will use.
        _string : str
            Number of threads the Random Forest will utilize.
        min : float
            Number of threads the Random Forest will utilize.
        max : float
            Number of threads the Random Forest will utilize.
        """
        from sympy import symbols
        from numpy import isclose

        if (min is None) and (max is None):
            _min, _max = np.percentile(_array, 1), np.percentile(_array, 99)
        else:
            _min, _max = min, max

        if isclose(_max, _min):
            raise ValueError("Trying to normalize {} will incur a divide by zero error.".format(_string))

        normalized_array = (_array - _min) / (_max - _min)

        x = symbols(_string)
        string = (x - _min) / (_max - _min)

        return [normalized_array, string, (_min, _max)]

    @staticmethod
    def tf_secant_h(x):
        return 2.0 / (tf.exp(x) + tf.exp(-1.0 * x))


def get_symbols(start, end):
    """
    Selection constructor function

    Athena uses a random forest from Scikit-Learn to select the best parameters. Use this constructor to create a Selection class.

    Parameters
    ----------
    start : int
        Number of estimators (decision trees) the Random Forest will use.
    end : int
        Number of threads the Random Forest will utilize.
    """
    return symbols(" ".join(["v{}".format(s) for s in list(range(start, end))]))


def SimpleSinusoidal(self, h, h_string):
    offset, variables = 3, 1
    eq = self.w[self._wc] * tf.sin(self.w[self._wc + 1] * h + self.w[self._wc + 2])

    if isinstance(h_string, str) or isinstance(h_string, Expr):
        syms = get_symbols(1, offset + variables + 1)
        equation_string = syms[0] * sin(syms[1] * syms[-1] + syms[2])
        parameters_list = [[h_string, syms[-1]]] + [[self.w[self._wc + i], syms[i]] for i in range(offset)]
    else:
        l = len(h_string["parameters"])
        syms = get_symbols(l + 1, offset + variables + l + 1)
        equation_string = syms[0] * sin(syms[1] * h_string["symbolic"] + syms[2])
        parameters_list = [h_string["parameters"][0]] + \
                          [[self.w[self._wc + i], syms[i]] for i in range(offset)] + \
                          h_string["parameters"][1:]

    return {"equation": eq, "parameters": parameters_list, "symbolic": equation_string, "offset": offset}


def SimpleSecant(self, h, h_string):
    offset, variables = 4, 1
    eq = self.w[self._wc] * (self.w[self._wc + 1] * (h - tf.abs(self.w[self._wc + 2])) ** 2 + self.w[self._wc + 3])

    if isinstance(h_string, str) or isinstance(h_string, Expr):
        syms = get_symbols(1, offset + variables + 1)
        equation_string = syms[0] * (syms[1] * (syms[-1] - abs(syms[2])) ** 2 + syms[3])
        parameters_list = [[h_string, syms[-1]]] + [[self.w[self._wc + i], syms[i]] for i in range(offset)]
    else:
        l = len(h_string["parameters"])
        syms = get_symbols(l + 1, offset + variables + l + 1)
        equation_string = syms[0] * (syms[1] * (h_string["symbolic"] - abs(syms[2])) ** 2 + syms[3])
        parameters_list = [h_string["parameters"][0]] + \
                          [[self.w[self._wc + i], syms[i]] for i in range(offset)] + \
                          h_string["parameters"][1:]

    return {"equation": eq, "parameters": parameters_list, "symbolic": equation_string, "offset": offset}


def Exponential(self, x, x_string):
    offset, variables = 2, 1
    eq = self.w[self._wc] * tf.exp(x * self.w[self._wc + 1])

    if isinstance(x_string, str) or isinstance(x_string, Expr):
        syms = get_symbols(1, offset + variables + 1)
        equation_string = syms[0] * exp(syms[-1] * syms[1])
        parameters_list = [[x_string, syms[-1]]] + [[self.w[self._wc + i], syms[i]] for i in range(offset)]
    else:
        l = len(x_string["parameters"])
        syms = get_symbols(l + 1, offset + variables + l + 1)
        equation_string = syms[0] * exp(x_string["symbolic"] * syms[1])
        parameters_list = [x_string["parameters"][0]] + \
                          [[self.w[self._wc + i], syms[i]] for i in range(offset)] + \
                          x_string["parameters"][1:]

    return {"equation": eq, "parameters": parameters_list, "symbolic": equation_string, "offset": offset}


def Bias(self):
    offset = 1
    eq = self.w[self._wc]
    b = symbols('b')
    equation_string = b
    parameters_list = [[self.w[self._wc], b]]
    return {"equation": eq, "parameters": parameters_list, "symbolic": equation_string, "offset": offset}


def FlexiblePower(self, x, x_string):
    offset, variables = 3, 1
    eq = self.w[self._wc] * ((x ** self.w[self._wc + 1]) + self.w[self._wc + 2])

    if isinstance(x_string, str) or isinstance(x_string, Expr):
        syms = get_symbols(1, offset + variables + 1)
        equation_string = syms[0] * (syms[-1] ** (syms[1] + 1.0) + syms[2])
        parameters_list = [[x_string, syms[-1]]] + [[self.w[self._wc + i], syms[i]] for i in range(offset)]
    else:
        l = len(x_string["parameters"])
        syms = get_symbols(l + 1, offset + variables + l + 1)
        equation_string = syms[0] * (x_string["symbolic"] ** (syms[1] + 1.0) + syms[2])
        parameters_list = [x_string["parameters"][0]] + \
                          [[self.w[self._wc + i], syms[i]] for i in range(offset)] + \
                          x_string["parameters"][1:]

    return {"equation": eq, "parameters": parameters_list, "symbolic": equation_string, "offset": offset}


def BipolarPolynomial(self, x, x_string, degree=1):
    offset, variables = 2 * degree, 1

    eq = 0
    d_index = 0
    for d in range(1, degree + 1):
        eq += self.w[self._wc + d_index] * ((x + self.w[self._wc + d_index + 1]) ** d)
        d_index += 2

    if isinstance(x_string, str) or isinstance(x_string, Expr):
        syms = get_symbols(1, offset + variables + 1)

        equation_string = 0
        d_index = 0
        for d in range(1, degree + 1):
            equation_string += syms[d_index] * ((syms[-1] + syms[d_index + 1]) ** d)
            d_index += 2

        parameters_list = [[x_string, syms[-1]]] + [[self.w[self._wc + i], syms[i]] for i in range(offset)]
    else:
        l = len(x_string["parameters"])
        syms = get_symbols(l + 1, offset + variables + l + 1)

        equation_string = 0
        d_index = 0
        for d in range(1, degree + 1):
            equation_string += syms[d_index] * ((x_string["symbolic"] + syms[d_index + 1]) ** d)
            d_index += 2

        parameters_list = [x_string["parameters"][0]] + \
                          [[self.w[self._wc + i], syms[i]] for i in range(offset)] + \
                          x_string["parameters"][1:]

    return {"equation": eq, "parameters": parameters_list, "symbolic": equation_string, "offset": offset}


def SimplePolynomial(self, x, x_string):
    offset, variables = 2, 1
    eq = tf.abs(self.w[self._wc]) * (x + tf.abs(self.w[self._wc + 1]))

    if isinstance(x_string, str) or isinstance(x_string, Expr):
        syms = get_symbols(1, offset + variables + 1)
        equation_string = abs(syms[0]) * (syms[-1] + abs(syms[1]))
        parameters_list = [[x_string, syms[-1]]] + [[self.w[self._wc + i], syms[i]] for i in range(offset)]
    else:
        l = len(x_string["parameters"])
        syms = get_symbols(l + 1, offset + variables + l + 1)
        equation_string = abs(syms[0]) * (x_string["symbolic"] + abs(syms[1]))
        parameters_list = [x_string["parameters"][0]] + \
                          [[self.w[self._wc + i], syms[i]] for i in range(offset)] + \
                          x_string["parameters"][1:]

    return {"equation": eq, "parameters": parameters_list, "symbolic": equation_string, "offset": offset}


def Logarithm(self, x, x_string):
    offset, variables = 2, 1
    eq = self.w[self._wc] * tf.log(x + self.w[self._wc + 1])

    if isinstance(x_string, str) or isinstance(x_string, Expr):
        syms = get_symbols(1, offset + variables + 1)
        equation_string = syms[0] * log(syms[-1] + syms[1])
        parameters_list = [[x_string, syms[-1]]] + [[self.w[self._wc + i], syms[i]] for i in range(offset)]
    else:
        l = len(x_string["parameters"])
        syms = get_symbols(l + 1, offset + variables + l + 1)
        equation_string = syms[0] * log(x_string["symbolic"] + syms[1])
        parameters_list = [x_string["parameters"][0]] + \
                          [[self.w[self._wc + i], syms[i]] for i in range(offset)] + \
                          x_string["parameters"][1:]

    return {"equation": eq, "parameters": parameters_list, "symbolic": equation_string, "offset": offset}


def Sinusoidal(self, x, x_string, h, h_string):
    offset, variables = 4, 2
    eq = self.w[self._wc] * (tf.sin(h * 2 * np.pi + self.w[self._wc + 1]) + self.w[self._wc + 2]) * (
        x + self.w[self._wc + 3])

    if isinstance(x_string, dict) or isinstance(h_string, dict):
        raise NotImplementedError("At this time multi-variable functions do not support composition.")

    syms = get_symbols(1, offset + variables + 1)
    equation_string = syms[0] * (sin(syms[-1] * 2 * pi + syms[1]) + syms[2]) * (syms[-2] + syms[3])
    parameters_list = [[x_string, syms[-2]], [h_string, syms[-1]]] + [[self.w[self._wc + i], syms[i]] for i in
                                                                      range(offset)]

    return {"equation": eq, "parameters": parameters_list, "symbolic": equation_string, "offset": offset}


def MultiPolynomial(self, *args):
    assert len(args) >= 2
    assert len(args) % 2 == 0

    variables = int(len(args) / 2)
    offset = 2 + 2 * variables

    variable_objects = []
    equation_objects = []
    for i, e in enumerate(args):
        if i % 2 == 0:
            variable_objects.append(e)
        else:
            equation_objects.append(e)

    for x_string in equation_objects:
        if isinstance(x_string, dict):
            raise NotImplementedError("At this time multi-variable functions do not support composition.")

    sub_eq = 1
    j = 2
    for x in variable_objects:
        sub_eq *= self.w[self._wc + j] * (x + self.w[self._wc + j + 1])
        j += 2
    eq = self.w[self._wc] * (sub_eq + self.w[self._wc + 1])

    syms = get_symbols(1, offset + variables + 1)

    sub_equation_string = 1
    j = 2
    for i in range(len(equation_objects)):
        sub_equation_string *= syms[j] * (syms[-1 * (i + 1)] + syms[j + 1])
        j += 2
    equation_string = syms[0] * (sub_equation_string + syms[1])

    parameters_list = []
    for i, j in enumerate(equation_objects):
        parameters_list.append([j, syms[-1 * (i + 1)]])
    parameters_list += [[self.w[self._wc + i], syms[i]] for i in range(offset)]

    return {"equation": eq, "parameters": parameters_list, "symbolic": equation_string, "offset": offset}


def WeightedAverage(self, *args):
    assert len(args) >= 2
    assert len(args) % 2 == 0

    variables = int(len(args) / 2)
    offset = variables

    variable_objects = []
    equation_objects = []
    for i, e in enumerate(args):
        if i % 2 == 0:
            variable_objects.append(e)
        else:
            equation_objects.append(e)

    for x_string in equation_objects:
        if isinstance(x_string, dict):
            raise NotImplementedError("At this time multi-variable functions do not support composition.")

    eq = 0
    weights = []

    for i, x in enumerate(variable_objects):
        weights.append(tf.abs(self.w[self._wc + i]))

    for i, x in enumerate(variable_objects):
        eq += (weights[i] / sum(weights)) * x

    syms = get_symbols(1, offset + variables + 1)

    equation_string = 0
    for i, j in enumerate(equation_objects):
        equation_string += abs(syms[i]) / sum([abs(x) for x in syms[:-1 * len(equation_objects)]]) * syms[-1 * (i + 1)]

    parameters_list = []
    for i, j in enumerate(equation_objects):
        parameters_list.append([j, syms[-1 * (i + 1)]])
    parameters_list += [[self.w[self._wc + i], syms[i]] for i in range(offset)]

    return {"equation": eq, "parameters": parameters_list, "symbolic": equation_string, "offset": offset}


def MultiDimensionalSecant(self, *args):
    assert len(args) >= 2
    assert len(args) % 2 == 0

    variables = int(len(args) / 2)
    offset = 2 + 3 * variables

    variable_objects = []
    equation_objects = []
    for i, e in enumerate(args):
        if i % 2 == 0:
            variable_objects.append(e)
        else:
            equation_objects.append(e)

    for x_string in equation_objects:
        if isinstance(x_string, dict):
            raise NotImplementedError("At this time multi-variable functions do not support composition.")

    sub_eq = 1
    j = 2
    for x in variable_objects:
        sub_eq *= Equation.tf_secant_h(self.w[self._wc + j] * x + self.w[self._wc + j + 1])
        j += 2
    eq = self.w[self._wc] * (sub_eq + self.w[self._wc + 1])

    syms = get_symbols(1, offset + variables + 1)

    sub_equation_string = 1
    j = 2
    for i in range(len(equation_objects)):
        sub_equation_string *= sech(syms[j] * syms[-1 * (i + 1)] + syms[j + 1])
        j += 2
    equation_string = syms[0] * (sub_equation_string + syms[1])

    parameters_list = []
    for i, j in enumerate(equation_objects):
        parameters_list.append([j, syms[-1 * (i + 1)]])
    parameters_list += [[self.w[self._wc + i], syms[i]] for i in range(offset)]

    return {"equation": eq, "parameters": parameters_list, "symbolic": equation_string, "offset": offset}
