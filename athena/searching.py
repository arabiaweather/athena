from athena.equations import *
from sympy import Symbol, lambdify, N
import numpy as np
from athena.framework import Framework
from athena.helpers import mean_confidence_interval
from sympy.core.cache import clear_cache

class RandomSearch:
	def __init__ (self, framework, search_length=100, equation_length=25, equations=None, starting_equations=None):
		self.search_length = search_length
		self.equation_length = equation_length
		self.framework = framework

		# If this is not none, then it is a list of args,kwargs (in the form of a dict), that
		# can be used to reconstruct an equation shape
		self.starting_equations = starting_equations

		if equations is not None:
			assert isinstance(equations, list)
			self.functions = equations
		else:
			self.functions = [SimpleSinusoidal, SimplePolynomial, BipolarPolynomial, FlexiblePower, Exponential,
			                  Logarithm, Sinusoidal, MultiPolynomial]

	def iteration (self, return_constituents=False):
		from random import choice
		from sklearn.metrics import mean_absolute_error, r2_score
		from scipy.stats import pearsonr
		from copy import copy

		fw = copy(self.framework).reset()

		training_targets = fw.dataset.training_targets
		testing_targets = fw.dataset.testing_targets

		from athena.model import AdditiveModel
		model = AdditiveModel(fw)

		if self.starting_equations is not None:
			for se in self.starting_equations:
				try:
					if choice(range(10)) >= 4:
						model.add(*se["args"], **se["kwargs"])
				except:
					continue
		else:
			model.add(Bias)

		i, j = 0, 0
		while j < self.equation_length:
			try:
				if choice([False, True]):
					model.add(choice(self.functions), choice(self.functions), choice(fw.dataset.get_columns()))
				elif choice([False, True]):
					model.add(choice(self.functions), choice(self.functions), choice(fw.dataset.get_columns()), choice(fw.dataset.get_columns()))
				else:
					model.add(choice(self.functions), choice(fw.dataset.get_columns()))

				j += 1
			except:
				continue

		self.searching_semaphore.acquire()

		try:
			fw.initialize(model, training_targets)
			fw.get_testing_predictions()
			for _ in range(int(fw.max_iters)):
				fw.run_learning_step()
		except:
			self.searching_semaphore.release()
			del fw
			return

		try:
			training_t = training_targets, fw.get_training_predictions()
			testing_t = testing_targets, fw.get_testing_predictions()

			training_error = mean_absolute_error(*training_t) / np.mean(training_targets) * 100.0
			testing_error = mean_absolute_error(*testing_t) / np.mean(testing_targets) * 100.0

			train_correlation, _ = pearsonr(*training_t)
			test_correlation, _ = pearsonr(*testing_t)

			training_r2 = r2_score(*training_t)
			testing_r2 = r2_score(*testing_t)

			if train_correlation > 0.05 and test_correlation > 0.05 and training_r2 > 0.05 and testing_r2 > 0.05:
				error = {
					"training_mae"    : training_error,
					"testing_mae"     : testing_error,

					"training_pearson": train_correlation,
					"testing_pearson" : test_correlation,

					"training_r2"     : training_r2,
					"testing_r2"      : testing_r2,
				}
			else:
				error = None

			equation = fw.produce_equation(constituents=return_constituents)
		except:
			error = None

		self.searching_semaphore.release()

		if error is not None:
			error["equation"] = copy(equation)
			error["reconstruction"] = copy(model.model_construction)
			self.equations.append(error)

		fw.session.close()
		del fw

	def search (self, metric="testing_r2", return_constituents=False):
		from tqdm import tqdm
		from threading import Thread, Semaphore
		from os import cpu_count
		import sys

		self.equations = []
		self.searching_threads = []
		self.searching_semaphore = Semaphore(value=cpu_count()*4)

		for iteration in range(self.search_length):
			self.searching_threads.append(
				Thread(target=self.iteration, kwargs={"return_constituents": return_constituents}))
			self.searching_threads[-1].start()

		for thread in tqdm(self.searching_threads):
			thread.join()

		sys.stdout.flush()

		self.equations = sorted(self.equations, key=lambda x: x[metric])
		if "pearson" in metric or "r2" in metric:
			self.equations = self.equations[::-1]

	def get_best_equations (self, k=1):
		if k != 0:
			return self.equations[:k]
		else:
			return self.equations


class GeneticSearch:
	def __init__ (self, fw: Framework, search_length, equation_length):
		self.fw = fw

		self.search_length = search_length
		self.equation_length = equation_length

		self.starting_equations = None
		self.iterations = 0

	def _remove_indices(self, x, indices):
		return [i for j, i in enumerate(x) if j not in indices]

	def iteration (self):
		self.iterations += 1

		while True:
			clear_cache()

			rs = RandomSearch(self.fw,
			                  search_length=self.search_length,
			                  equation_length=self.equation_length,
			                  starting_equations=self.starting_equations)

			rs.search(return_constituents=True)
			equation = rs.get_best_equations(k=0)

			clear_cache()

			if len(equation) == 0:
				print("Iteration produced bad results, retrying.")
			else:
				print("Iteration produced {} equations.".format(len(equation)))
				equation = equation[0]
				break

		biases, equations = 0, []
		for i, cnst in enumerate(equation["equation"]):
			if len(cnst.atoms(Symbol)) == 0:
				biases += cnst
			else:
				equations += [cnst]

		equation["equation"] = [biases] + equations

		eq_df = []

		for i, cnst in enumerate(equation["equation"]):
			if not hasattr(cnst, 'atoms'): continue
			variables = cnst.atoms(Symbol)

			substitutions = [self.fw.dataset.training_df[self.fw.dataset.inverse_parameters_map[str(v)]].values for v in variables]
			function = lambdify(tuple(variables), cnst, "numpy")
			_contribution = np.abs(function(*tuple(substitutions))) / self.fw.dataset.training_targets * 100.0
			contribution = mean_confidence_interval(_contribution, 0.99)

			eq_df.append({"constituent" : str(cnst),
			              "contribution": int(round(contribution[0]))})

		equation["constituents"] = eq_df

		indexes_to_remove = []

		for i, j in enumerate(equation["constituents"]):
			if j["contribution"] < 1:
				indexes_to_remove.append(i)

		equation["constituents"] = self._remove_indices(equation["constituents"], indexes_to_remove)
		equation["reconstruction"] = self._remove_indices(equation["reconstruction"], indexes_to_remove)
		equation["equation"] = self._remove_indices(equation["equation"], indexes_to_remove)

		self.starting_equations = equation["reconstruction"]

		return equation
