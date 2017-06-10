from athena.equations import *

class RandomSearch:
	def __init__(self, framework, search_length=100, equation_length=25, equations=None):
		self.search_length = search_length
		self.equation_length = equation_length
		self.framework = framework

		if equations is not None:
			assert isinstance(equations, list)
			self.functions = equations
		else:
			self.functions = [SimpleSinusoidal, BipolarPolynomial, FlexiblePower]

	def iteration(self):
		from random import choice
		from sklearn.metrics import mean_absolute_error, r2_score
		from scipy.stats import pearsonr
		from copy import copy

		self.searching_semaphore.acquire()

		fw = copy(self.framework).reset()

		training_targets = fw.dataset.training_targets
		testing_targets = fw.dataset.testing_targets

		from athena.model import AdditiveModel
		model = AdditiveModel(fw)

		# ==========================================================================================
		# Random model definition
		# ==========================================================================================
		model.add(Bias)
		for j in range(self.equation_length):
			if choice([False, True]):
				if choice([False, True]):
					model.add(choice(self.functions), choice(self.functions), choice(fw.dataset.get_columns()))
				else:
					model.add(choice(self.functions), choice(fw.dataset.get_columns()))

		fw.initialize(model, training_targets)

		fw.get_testing_predictions()

		for _ in range(int(fw.max_iters)):
			fw.run_learning_step()

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

		except Exception as e:
			error = None

		equation = fw.produce_equation()
		fw.session.close()
		del fw

		self.searching_semaphore.release()

		if error is not None:
			error["equation"] = equation
			self.equations.append(error)

	def search(self, metric="testing_r2"):
		from tqdm import tqdm
		from threading import Thread, Semaphore
		from os import cpu_count

		self.equations = []
		self.searching_threads = []
		self.searching_semaphore = Semaphore(value=cpu_count())

		for iteration in range(self.search_length):
			self.searching_threads.append(Thread(target=self.iteration))
			self.searching_threads[-1].start()

		for thread in tqdm(self.searching_threads):
			thread.join()

		self.equations = sorted(self.equations, key=lambda x: x[metric])
		if "pearson" in metric or "r2" in metric:
			self.equations = self.equations[::-1]

	def get_best_equations(self, k=1):
		return self.equations[:k]