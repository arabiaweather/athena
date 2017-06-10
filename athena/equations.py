import tensorflow as tf
import numpy as np
from sympy import symbols, sin, pi, exp, Expr

class Equation:
	def __init__ (self):
		# TODO: the number of Tensorflow variables should not be fixed but instead should change dynamically as the model grows
		self.w = tf.Variable(tf.random_normal([100], mean=0, stddev=1), name='coefficients')

		self._wc = int(0)
		self.equations = []
		self.datasets = {}

	def produce_equations (self, session):
		from tensorflow.python.framework.ops import Tensor
		from sympy.core.symbol import Symbol

		equation = []
		for e in self.equations:
			for sub in e["substitutions"]:
				if isinstance(sub[0], Tensor): sub[0] = session.run(sub[0])

				assert isinstance(sub[1], Symbol)
				e["equation-string"] = e["equation-string"].subs(sub[1], sub[0])

			equation.append(e["equation-string"])

		return equation

	def set_weights (self, new_weights):
		return self.w.assign(new_weights)

	# TODO: change min and max names so they don't shadow the built in functions
	@staticmethod
	def normalize (_array, _string, min=None, max=None):
		from sympy import symbols
		from math import isclose

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


def _get_symbols (start, end):
	return symbols(" ".join(["v{}".format(s) for s in list(range(start, end))]))


def SimpleSinusoidal (self, h, h_string):
	offset, variables = 3, 1
	eq = self.w[self._wc] * tf.sin(self.w[self._wc + 1] * h * 2 * np.pi + self.w[self._wc + 2])

	if isinstance(h_string, str) or isinstance(h_string, Expr):
		syms = _get_symbols(1, offset + variables + 1)
		equation_string = syms[0] * sin(syms[1] * syms[-1] * 2 * pi + syms[2])
		parameters_list = [[h_string, syms[-1]]] + [[self.w[self._wc + i], syms[i]] for i in range(offset)]
	else:
		l = len(h_string["parameters"])
		syms = _get_symbols(l + 1, offset + variables + l + 1)
		equation_string = syms[0] * sin(syms[1] * h_string["symbolic"] * 2 * pi + syms[2])
		parameters_list = [h_string["parameters"][0]] + \
		                  [[self.w[self._wc + i], syms[i]] for i in range(offset)] + \
		                  h_string["parameters"][1:]


	return {"equation": eq, "parameters": parameters_list, "symbolic": equation_string, "offset": offset}


def Exponential (self, x, x_string):
	offset, variables = 2, 1
	eq = self.w[self._wc] * tf.exp(x * self.w[self._wc + 1])

	if isinstance(x_string, str) or isinstance(x_string, Expr):
		syms = _get_symbols(1, offset + variables + 1)
		equation_string = syms[0] * exp(syms[-1] * syms[1])
		parameters_list = [[x_string, syms[0]]] + [[self.w[self._wc + i], syms[i]] for i in range(offset)]
	else:
		l = len(x_string["parameters"])
		syms = _get_symbols(l + 1, offset + variables + l + 1)
		equation_string = syms[0] * exp(x_string["symbolic"] * syms[1])
		parameters_list = [x_string["parameters"][0]] + \
		                  [[self.w[self._wc + i], syms[i]] for i in range(offset)] + \
		                  x_string["parameters"][1:]

	return {"equation": eq, "parameters": parameters_list, "symbolic": equation_string, "offset": offset}


def Bias (self):
	offset = 1
	eq = self.w[self._wc]
	b = symbols('b')
	equation_string = b
	parameters_list = [[self.w[self._wc], b]]
	return {"equation": eq, "parameters": parameters_list, "symbolic": equation_string, "offset": offset}


def FlexiblePower (self, x, x_string):
	offset, variables = 3, 1
	eq = self.w[self._wc] * (tf.pow(x, self.w[self._wc + 1] + 1) + self.w[self._wc + 2])

	if isinstance(x_string, str) or isinstance(x_string, Expr):
		syms = _get_symbols(1, offset + variables + 1)
		equation_string = syms[0] * (syms[-1] ** (syms[1] + 1.0) + syms[2])
		parameters_list = [[x_string, syms[-1]]] + [[self.w[self._wc + i], syms[i]] for i in range(offset)]
	else:
		l = len(x_string["parameters"])
		syms = _get_symbols(l + 1, offset + variables + l + 1)
		equation_string = syms[0] * (x_string["symbolic"] ** (syms[1] + 1.0) + syms[2])
		parameters_list = [x_string["parameters"][0]] + \
		                  [[self.w[self._wc + i], syms[i]] for i in range(offset)] + \
		                  x_string["parameters"][1:]

	return {"equation": eq, "parameters": parameters_list, "symbolic": equation_string, "offset": offset}


def BipolarPolynomial (self, x, x_string):

	offset, variables = 2, 1
	eq = self.w[self._wc] * (x + self.w[self._wc + 1])

	if isinstance(x_string, str) or isinstance(x_string, Expr):
		syms = _get_symbols(1, offset + variables + 1)
		equation_string = syms[0] * (syms[-1] + syms[1])
		parameters_list = [[x_string, syms[-1]]] + [[self.w[self._wc + i], syms[i]] for i in range(offset)]
	else:
		l = len(x_string["parameters"])
		syms = _get_symbols(l + 1, offset + variables + l + 1)
		equation_string = syms[0] * (x_string["symbolic"] + syms[1])
		parameters_list = [x_string["parameters"][0]] + \
		                  [[self.w[self._wc + i], syms[i]] for i in range(offset)] + \
		                  x_string["parameters"][1:]

	return {"equation": eq, "parameters": parameters_list, "symbolic": equation_string, "offset": offset}


# TODO: fix this function to conform to the new function standards (as in the functions above)
def HigherOrderPolynomial (self, x, x_string, degree=3):
	assert isinstance(degree, int)
	from string import ascii_lowercase
	alphabet = list(ascii_lowercase)
	offset = degree * 2
	eq = 0
	equation_string = 0
	list_of_symbols = symbols(' '.join(alphabet))
	z = list_of_symbols[-1]
	parameters_list = [[x_string, z]]
	d = 0
	while d < offset:
		eq += self.w[self._wc + d] * (tf.pow(x, (d / 2) + 1) + self.w[self._wc + d + 1])
		equation_string += list_of_symbols[d] * (z ** ((d / 2) + 1) + list_of_symbols[d + 1])
		parameters_list += [[self.w[self._wc + d], list_of_symbols[d]],
		                    [self.w[self._wc + d + 1], list_of_symbols[d + 1]]]
		d += 2
	function_definition(self, parameters_list, offset, equation_string)
	return eq


# TODO: fix this function to conform to the new function standards (as in the functions above)
def Sinusoidal (self, x, x_string, h, h_string):
	offset, variables = 4, 2
	eq = self.w[self._wc] * (tf.sin(h * 2 * np.pi + self.w[self._wc + 1]) + self.w[self._wc + 2]) * (x + self.w[self._wc + 3])

	if isinstance(x_string, str) and isinstance(h_string, str):
		syms = _get_symbols(1, offset + variables + 1)
		equation_string = syms[0] * (sin(syms[-1] * 2 * pi + syms[1]) + syms[2]) * (syms[-2] + syms[3])
		parameters_list = [[x_string, syms[-2]], [h_string, syms[-1]]] + [[self.w[self._wc + i], syms[i]] for i in range(offset)]
	elif (not isinstance(x_string, str)) and (not isinstance(h_string, str)):
		raise Exception()
	else:
		raise Exception()

	return {"equation": eq, "parameters": parameters_list, "symbolic": equation_string, "offset": offset}