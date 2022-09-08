from __future__ import print_function
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class Framework:
	def __init__ (self, framework_parameters=None):
		"""
        Selection constructor function

        Athena uses a random forest from Scikit-Learn to select the best parameters. Use this constructor to create a Selection class.

        Parameters
        ----------
        framework_parameters : int
            Number of estimators (decision trees) the Random Forest will use.
        """
		from athena.equations import Equation

		# ==========================================================================================
		# Here are the default framework hyper-parameters that can be overriden during class init
		# TODO: scientifically determine the best default parameters for the majority of cases
		# ==========================================================================================
		fp = {
			"starting_lr": 0.05,
 			"max_iterations": int(1e4),
 			"momentum": 0.99,
 			"optimizer": tf.train.AdamOptimizer
		}

		if framework_parameters is not None:
			for f, p in framework_parameters.items():
				if f in fp:
					fp[f] = p
				else:
					raise Exception("Unknown parameter with name {} given to framework initializer.".format(f))

		self.starting_lr, self.max_iters, self.momentum = fp["starting_lr"], fp["max_iterations"], fp["momentum"]
		self.tf_optimizer = fp["optimizer"]

		self.tf_graph = tf.Graph()
		self.session = tf.Session(graph=self.tf_graph)

		self.eqn = Equation(self.tf_graph)
		self.eqn_test = Equation(self.tf_graph)
		self.dataset = None
		self.model = None

	def reset (self):
		"""
        Selection constructor function

        Athena uses a random forest from Scikit-Learn to select the best parameters. Use this constructor to create a Selection class.
        """
		from athena.equations import Equation

		self.model = None
		self.tf_graph = tf.Graph()
		self.session = tf.Session(graph=self.tf_graph)

		self.eqn = Equation(self.tf_graph)
		self.eqn_test = Equation(self.tf_graph)

		return self

	def initialize (self, model, targets=None, loss_function="mse"):
		"""
        Selection constructor function

        Athena uses a random forest from Scikit-Learn to select the best parameters. Use this constructor to create a Selection class.

        Parameters
        ----------
        model : int
            Number of estimators (decision trees) the Random Forest will use.
        targets : int
            Number of threads the Random Forest will utilize.
        loss_function : int
            Number of threads the Random Forest will utilize.
        """
		from athena.model import Model

		assert isinstance(loss_function, str)
		self.loss_function = loss_function

		if targets is None:
			targets = self.dataset.training_targets

		assert isinstance(model, Model)
		predictions = model.get_training_equation()

		assert self.dataset is not None

		self.model = model
		self.training_targets = targets

		# TODO: fix the non deterministic reduce function when on gpu or multi cpu
		# see: https://www.twosigma.com/insights/a-workaround-for-non-determinism-in-tensorflow

		with self.tf_graph.as_default():
			if self.loss_function == "mse":
				training_loss = tf.reduce_mean(tf.square(predictions - targets))
			elif self.loss_function == "r2":
				numerator = tf.reduce_sum(((targets - predictions) ** 2.0), axis=0)
				denominator = tf.reduce_sum((targets - tf.reduce_mean(targets, axis=0)) ** 2.0, axis=0)
				training_loss = -1.0 * tf.reduce_mean(1.0 - (numerator / denominator))
			elif self.loss_function == "pearson":
				mx = tf.reduce_mean(targets)
				my = tf.reduce_mean(predictions)
				xm, ym = tf.subtract(targets, mx), tf.subtract(predictions, my)
				r_num = tf.tensordot(xm, ym, axes=1)
				r_den = tf.sqrt(tf.reduce_sum(xm ** 2.0) * tf.reduce_sum(ym ** 2.0))
				training_loss = -1.0 * r_num / r_den
			else:
				raise ValueError("{} is not a valid loss function.".format(loss_function))

			global_step = tf.Variable(0, trainable=False)
			self.learning_rate = tf.train.exponential_decay(self.starting_lr, global_step, self.max_iters, self.momentum, staircase=True)
			self.learning_step = self.tf_optimizer(self.learning_rate).minimize(training_loss, global_step=global_step)

			tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
			self.init = tf.global_variables_initializer()
			self.session.run(self.init)

		return self

	def add_dataset (self, dataset):
		"""
        Selection constructor function

        Athena uses a random forest from Scikit-Learn to select the best parameters. Use this constructor to create a Selection class.

        Parameters
        ----------
        dataset : int
            Number of estimators (decision trees) the Random Forest will use.
        """
		from athena.dataset import Dataset
		assert isinstance(dataset, Dataset)
		self.dataset = dataset

	def run (self, x):
		"""
        Selection constructor function

        Athena uses a random forest from Scikit-Learn to select the best parameters. Use this constructor to create a Selection class.

        Parameters
        ----------
        x : ?
            Number of estimators (decision trees) the Random Forest will use.
        """
		return self.session.run(x)

	def run_learning_step (self):
		"""
        Selection constructor function

        Athena uses a random forest from Scikit-Learn to select the best parameters. Use this constructor to create a Selection class.
        """
		self.run(self.learning_step)

	def before_training_checks (self):
		# TODO: insert assertions here to check the state of the framework before training is allowed to begin
		pass

	def after_training_checks (self):
		# TODO: insert assertions here to check the state of the framework (and training results) after training has completed
		pass

	def train (self):
		"""
        Selection constructor function

        Athena uses a random forest from Scikit-Learn to select the best parameters. Use this constructor to create a Selection class.
        """
		self.before_training_checks()
		for _ in range(self.max_iters):
			self.run_learning_step()
		self.after_training_checks()

	def produce_equation (self, constituents=False):
		"""
        Selection constructor function

        Athena uses a random forest from Scikit-Learn to select the best parameters. Use this constructor to create a Selection class.

        Parameters
        ----------
        constituents : bool
            Number of estimators (decision trees) the Random Forest will use.
        """
		from sympy import init_printing
		init_printing()
		equations = self.eqn.produce_equations(self.session)
		if constituents: return equations
		return self.model.consolidate(equations)

	def get_training_predictions (self):
		"""
        Selection constructor function

        Athena uses a random forest from Scikit-Learn to select the best parameters. Use this constructor to create a Selection class.
        """
		return self.run(self.model.get_training_equation())

	def get_testing_predictions (self):
		"""
        Selection constructor function

        Athena uses a random forest from Scikit-Learn to select the best parameters. Use this constructor to create a Selection class.
        """
		# copy over weights from training equation to testing equation first before running the equation
		self.run(self.eqn_test.set_weights(self.run(self.eqn.w)))
		return self.run(self.model.get_testing_equation())

	def close_session (self):
		"""
        Selection constructor function

        Athena uses a random forest from Scikit-Learn to select the best parameters. Use this constructor to create a Selection class.
        """
		self.session.close()
		del self.tf_graph
		del self.session
