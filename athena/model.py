class Model:
	def __init__ (self, framework):
		from athena.framework import Framework
		assert isinstance(framework, Framework)

		self.learning_equations = None
		self.training_equation = framework.eqn
		self.testing_equation = framework.eqn_test
		self.datasets = framework.dataset

	def get_training_equation (self):
		assert "training" in self.learning_equations
		return self.learning_equations["training"]

	def get_testing_equation (self):
		assert "testing" in self.learning_equations
		return self.learning_equations["testing"]

	@staticmethod
	def function_definition (equation_class, parameters_list, equation_string):
		equation_class.equations.append({
			"equation-string": equation_string,
			"substitutions"  : parameters_list,
		})

	@staticmethod
	def add_offset (equation_class, offset):
		equation_class._wc += offset



class AdditiveModel(Model):
	def add (self, *args, **kwargs):
		from types import FunctionType
		function, parameter = None, None

		# Parameter can either be a string (name of column in self.datasets), or it can
		# be a list of strings (multiple parameters with names in self.datasets too)
		if "parameter" in kwargs and kwargs["parameter"] is not None:
			parameter = kwargs["parameter"]
			assert isinstance(parameter, str) or isinstance(parameter, list)

		# Function can either be a function (from the Equations file), or it can
		# be a list of functions (also from the Equations file): this is for function
		# composition, and functions are composed in ascending order
		# eg: [f1, f2, ..., f3] will mean f1(f2(f3(parameters)))
		if "function" in kwargs and kwargs["function"] is not None:
			function = kwargs["function"]
			assert isinstance(function, FunctionType) or isinstance(function, list)

		if function is None:
			function = []
			for f in args:
				if isinstance(f, FunctionType):
					function.append(f)

			assert len(function) > 0

		if parameter is None:
			parameter = []
			for p in args:
				if isinstance(p, str):
					parameter.append(p)

		if self.learning_equations is None:
			self.learning_equations = {"training": 0, "testing": 0}

		if isinstance(function, FunctionType):
			f = function
		else:
			f = function[-1]

		for learning_case in ["training", "testing"]:
			if learning_case == "training":
				_eqn = self.training_equation
			else:
				_eqn = self.testing_equation

			if parameter is not None and isinstance(parameter, str):
				result = f(_eqn, *self.datasets.get(learning_case)[parameter], **kwargs)

			elif parameter is not None and isinstance(parameter, dict):
				result = f(_eqn, parameter["equation"], parameter["symbolic"], **kwargs)

			elif parameter is not None and isinstance(parameter, list) and len(parameter) > 0:
				inputs = []
				for p in parameter:
					if isinstance(p, str):
						inputs += list(self.datasets.get(learning_case)[p])
					elif isinstance(p, dict):
						inputs += [p["equation"], p["symbolic"]]
					else:
						raise Exception()

				result = f(_eqn, *inputs, **kwargs)

			else:
				result = f(_eqn, **kwargs)

			self.learning_equations[learning_case] += result["equation"]
			self.add_offset(_eqn, result["offset"])

			if len(function) == 1:
				self.function_definition(_eqn, result["parameters"], result["symbolic"])
			else:
				self.add_residual(_eqn, function[:-1], result, learning_case, **kwargs)

	def add_residual(self, _eqn, function, parameter, learning_case, **kwargs):
		f = function[-1]

		if parameter is not None and isinstance(parameter, str):
			raise Exception()

		elif parameter is not None and isinstance(parameter, dict):
			result = f(_eqn, parameter["equation"], parameter, **kwargs)

		elif parameter is not None and isinstance(parameter, list):
			raise Exception()

		else:
			result = f(_eqn, **kwargs)

		self.learning_equations[learning_case] += result["equation"]
		self.add_offset(_eqn, result["offset"])

		if len(function) == 1:
			self.function_definition(_eqn, result["parameters"], result["symbolic"])
		else:
			self.add_residual(_eqn, function[:-1], parameter, learning_case, **kwargs)

	@staticmethod
	def consolidate (list_of_equations: list):
		assert isinstance(list_of_equations, list)
		return sum(list_of_equations)


class MultiplicativeModel(Model):
	def add (self, function, parameter=None):
		if parameter is not None:
			assert isinstance(parameter, str)

		if self.learning_equations is None:
			self.learning_equations = {"training": 1, "testing": 1}

		for learning_case in ["training", "testing"]:
			if learning_case == "training":
				_eqn = self.training_equation
			else:
				_eqn = self.testing_equation

			if parameter is not None:
				self.learning_equations[learning_case] *= function(_eqn, *self.datasets.get(learning_case)[parameter])
			else:
				self.learning_equations[learning_case] *= function(_eqn)

	@staticmethod
	def consolidate (list_of_equations: list):
		assert isinstance(list_of_equations, list)
		x = list_of_equations[0]
		for y in list_of_equations[1:]: x *= y
		return x
