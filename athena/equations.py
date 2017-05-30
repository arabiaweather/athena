import tensorflow as tf
import numpy as np

class Equation:
	def __init__ (self):
		# TODO: the number of Tensorflow variables should not be fixed but instead should change dynamically as the model grows
		self.w = tf.Variable(tf.random_normal([100], mean=0, stddev=0.1), name='coefficients')
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


from sympy import symbols, sin, cos, pi, exp


def function_definition (self, parameters_list, offset, equation_string):
	self.equations.append({
		"equation-string": equation_string,
		"substitutions"  : parameters_list,
	})
	self._wc += offset


def Sinusoidal (self, x, x_string, h, h_string):
	offset = 4
	a, b, c, d, p, q = symbols('a b c d p q')
	eq = self.w[self._wc] * (tf.sin(h * 2 * np.pi + self.w[self._wc + 1]) + self.w[self._wc + 2]) * (x + self.w[self._wc + 3])
	equation_string = a * (sin(q * 2*pi + b) + c) * (p + d)
	parameters_list = [[x_string, p], [h_string, q], [self.w[self._wc], a], [self.w[self._wc + 1], b], [self.w[self._wc + 2], c], [self.w[self._wc + 3], d]]
	function_definition(self, parameters_list, offset, equation_string)
	return eq


def SimpleSinusoidal (self, h, h_string):
	offset = 3
	eq = self.w[self._wc] * tf.sin(self.w[self._wc + 1] * h * 2 * np.pi + self.w[self._wc + 2])
	a, b, c, p = symbols('a b c p')
	equation_string = a * sin(b * p * 2 * pi + c)
	parameters_list = [[h_string, p], [self.w[self._wc], a], [self.w[self._wc + 1], b], [self.w[self._wc + 2], c]]
	function_definition(self, parameters_list, offset, equation_string)
	return eq


def SinusoidalBiasCorrection (self, x, x_string, h, h_string):
	offset = 4
	a, b, c, d, p, q = symbols('a b c d p q')
	eq = tf.pow(self.w[self._wc], 2.0) * (tf.pow(self.w[self._wc + 1], 2.0) * x + self.w[self._wc + 2] * tf.sin(h * 2 * np.pi + self.w[self._wc + 3]))
	equation_string = a ** 2 * (b ** 2 * p + c * sin(q * 2 * pi + d))
	parameters_list = [[x_string, p], [h_string, q], [self.w[self._wc], a], [self.w[self._wc + 1], b], [self.w[self._wc + 2], c], [self.w[self._wc + 3], d]]
	function_definition(self, parameters_list, offset, equation_string)
	return eq


def MultipleSinusoidals (self, h, h_string):
	offset = 4
	a, b, c, d, p = symbols('a b c d p')
	eq = self.w[self._wc] * tf.sin(h * 2 * np.pi + self.w[self._wc + 1]) + self.w[self._wc + 2] * tf.cos(h * 2 * np.pi + self.w[self._wc + 3])
	equation_string = a * sin(p * 2 * pi + b) + c * cos(p * 2 * pi + d)
	parameters_list = [[h_string, p], [self.w[self._wc], a], [self.w[self._wc + 1], b], [self.w[self._wc + 2], c], [self.w[self._wc + 3], d]]
	function_definition(self, parameters_list, offset, equation_string)
	return eq


def Exponential (self, x, x_string):
	offset = 2
	a, b, p = symbols('a b p')
	eq = self.w[self._wc] * tf.exp(x * self.w[self._wc + 1])
	equation_string = a * exp(p * b)
	parameters_list = [[x_string, p], [self.w[self._wc], a], [self.w[self._wc + 1], b]]
	function_definition(self, parameters_list, offset, equation_string)
	return eq


def SimplePolynomial (self, x, x_string):
	offset = 1
	a, p = symbols('a p')
	eq = tf.pow(self.w[self._wc], 2.0) * x
	equation_string = pow(a, 2) * p
	parameters_list = [[x_string, p], [self.w[self._wc], a]]
	function_definition(self, parameters_list, offset, equation_string)
	return eq


def Polynomial (self, x, x_string):
	offset = 2
	a, b, p = symbols('a b p')
	eq = tf.pow(self.w[self._wc], 2.0) * (x + self.w[self._wc + 1])
	equation_string = (a ** 2) * (p + b)
	parameters_list = [[x_string, p], [self.w[self._wc], a], [self.w[self._wc + 1], b]]
	function_definition(self, parameters_list, offset, equation_string)
	return eq


def MultiplePolynomials (self, x, x_string, y, y_string):
	offset = 2
	a, b, p, q = symbols('a b p q')
	eq = self.w[self._wc] * (x * y + self.w[self._wc + 1])
	equation_string = a * (p * q + b)
	parameters_list =  [[x_string, p], [y_string, q], [self.w[self._wc], a], [self.w[self._wc + 1], b]]
	function_definition(self, parameters_list, offset, equation_string)
	return eq


def Bias (self):
	offset = 1
	eq = self.w[self._wc]
	b = symbols('b')
	equation_string = b
	parameters_list = [[self.w[self._wc], b]]
	function_definition(self, parameters_list, offset, equation_string)
	return eq


def FlexiblePower (self, x, x_string):
	offset = 3
	eq = self.w[self._wc] * (tf.pow(x, self.w[self._wc + 1] + 1) + self.w[self._wc + 2])
	a, b, c, p = symbols('a b c p')
	equation_string = a * (p ** (b + 1.0) + c)
	parameters_list = [[x_string, p], [self.w[self._wc], a], [self.w[self._wc + 1], b], [self.w[self._wc + 2], c]]
	function_definition(self, parameters_list, offset, equation_string)
	return eq


def BipolarPolynomial (self, x, x_string):
	offset = 2
	eq = self.w[self._wc] * (x + self.w[self._wc + 1])
	a, b, p = symbols('a b p')
	equation_string = a * (p + b)
	parameters_list = [[x_string, p], [self.w[self._wc], a], [self.w[self._wc + 1], b]]
	function_definition(self, parameters_list, offset, equation_string)
	return eq

def HigherOrderPolynomial (self, x, x_string, degree=3):
	assert isinstance(degree, int)
	from string import ascii_lowercase
	alphabet = list(ascii_lowercase)
	offset = degree*2
	eq = 0
	equation_string = 0
	list_of_symbols = symbols(' '.join(alphabet))
	z = list_of_symbols[-1]
	parameters_list = [[x_string, z]]
	d = 0
	while d < offset:
		eq += self.w[self._wc + d] * (tf.pow(x, (d/2)+1) + self.w[self._wc + d+1])
		equation_string += list_of_symbols[d] * (z**((d/2)+1) + list_of_symbols[d+1])
		parameters_list += [[self.w[self._wc + d], list_of_symbols[d]], [self.w[self._wc + d+1], list_of_symbols[d+1]]]
		d += 2
	function_definition(self, parameters_list, offset, equation_string)
	return eq