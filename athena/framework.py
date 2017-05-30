from athena.model import Model
from athena.dataset import Dataset

class Framework:
	def __init__(self, framework_parameters=None):
		from athena.equations import Equation
		import tensorflow

		# ==========================================================================================
		# Here are the default framework hyper-parameters that can be overriden during class init
		# TODO: scientifically determine the best default parameters for the majority of cases
		# ==========================================================================================
		fp = {
			"starting_lr": 0.05,
			"max_iterations": int(1e4),
			"momentum": 0.99,
			"optimizer": tensorflow.train.AdamOptimizer,
		}

		if framework_parameters is not None:
			for f,p in framework_parameters.items():
				if f in fp:
					fp[f] = p
				else:
					raise Exception("Unknown parameter with name {} given to framework initializer.".format(f))

		self.starting_lr, self.max_iters, self.momentum = fp["starting_lr"], fp["max_iterations"], fp["momentum"]
		self.tf_optimizer = fp["optimizer"]
		self.eqn = Equation()
		self.eqn_test = Equation()
		self.dataset = None
		self.model = None


	def initialize(self, model: Model, targets):
		assert isinstance(model, Model)
		predictions = model.get_training_equation()

		assert self.dataset is not None

		import tensorflow

		self.model = model

		# TODO: fix the non deterministic reduce function when on gpu or multi cpu
		# see: https://www.twosigma.com/insights/a-workaround-for-non-determinism-in-tensorflow
		training_loss = tensorflow.reduce_mean(tensorflow.square(predictions - targets))

		global_step = tensorflow.Variable(0, trainable=False)

		self.learning_rate = tensorflow.train.exponential_decay(self.starting_lr, global_step, self.max_iters, self.momentum, staircase=True)
		self.learning_step = self.tf_optimizer(self.learning_rate).minimize(training_loss, global_step=global_step)

		tensorflow.group(tensorflow.global_variables_initializer(), tensorflow.local_variables_initializer())
		
		self.session = tensorflow.Session()
		self.init = tensorflow.global_variables_initializer()
		self.session.run(self.init)

	def add_dataset(self, dataset: Dataset):
		assert isinstance(dataset, Dataset)
		self.dataset = dataset

	def run(self, x):
		return self.session.run(x)

	def run_learning_step(self):
		self.run(self.learning_step)

	def before_training_checks(self):
		# TODO: insert assertions here to check the state of the framework before training is allowed to begin
		pass

	def after_training_checks(self):
		# TODO: insert assertions here to check the state of the framework (and training results) after training has completed
		pass

	def train(self):
		self.before_training_checks()
		for _ in range(self.max_iters):
			self.run_learning_step()
		self.after_training_checks()

	def produce_equation(self):
		from sympy import init_printing
		init_printing()
		equations = self.eqn.produce_equations(self.session)
		return self.model.consolidate(equations)

	def get_training_predictions(self):
		return self.run(self.model.get_training_equation())

	def get_testing_predictions(self):
		# copy over weights from training equation to testing equation first before running the equation
		self.run(self.eqn_test.set_weights(self.run(self.eqn.w)))
		return self.run(self.model.get_testing_equation())
