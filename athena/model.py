from athena.framework import Framework


class Model:
	def __init__ (self, framework: Framework):
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


class AdditiveModel(Model):
	def add (self, function, parameter=None, **kwargs):
		if parameter is not None:
			assert isinstance(parameter, str) or isinstance(parameter, list)

		if self.learning_equations is None:
			self.learning_equations = {"training": 0, "testing": 0}

		for learning_case in ["training", "testing"]:
			if learning_case == "training":
				_eqn = self.training_equation
			else:
				_eqn = self.testing_equation

			if parameter is not None and isinstance(parameter, str):
				self.learning_equations[learning_case] += \
					function(_eqn, *self.datasets.get(learning_case)[parameter], **kwargs)

			elif parameter is not None and isinstance(parameter, list):
				inputs = []
				for p in parameter: inputs += list(self.datasets.get(learning_case)[p])
				self.learning_equations[learning_case] += function(_eqn, *inputs, **kwargs)

			else:
				self.learning_equations[learning_case] += function(_eqn, **kwargs)

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
