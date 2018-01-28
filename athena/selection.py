class Selection:
	def __init__(self, n_estimators=250, n_jobs=-1):
		"""
		Selection constructor function

		Athena uses a random forest from Scikit-Learn to select the best parameters. Use this constructor to create a Selection class.

		Parameters
		----------
		n_estimators : int
			Number of estimators (decision trees) the Random Forest will use.
		n_jobs : int
			Number of threads the Random Forest will utilize.
		"""
		from sklearn.ensemble import ExtraTreesRegressor
		self.clf = ExtraTreesRegressor(n_estimators=n_estimators, n_jobs=n_jobs)

	def train(self, data_frame, columns, target_column_name):
		"""
		Train a Selection class

		Train the current Selection class to find the best columns to be used that will predict the target column.

		Parameters
		----------
		data_frame : pandas.DataFrame
		columns : list
		target_column_name : str
		"""
		from pandas import DataFrame

		assert isinstance(data_frame, DataFrame)
		assert isinstance(columns, list)
		assert isinstance(target_column_name, str)

		self.columns = columns
		self.clf.fit(data_frame[columns].values, data_frame[target_column_name].values)

	def select(self, n, fixed_features=None):
		"""
		Select features from a Selection class

		Select n number of features (columns in the specified dataframe). The train function must always be called before calling this function.

		Parameters
		----------
		n : pandas.DataFrame
		fixed_features : list
		"""
		assert isinstance(n, int)

		if fixed_features is None:
			fixed_features = []

		from numpy import argsort

		importance = self.clf.feature_importances_
		indices = argsort(importance)[::-1]
		best_features = []

		f = 0
		while len(best_features) < n:
			if not (self.columns[indices[f]] in fixed_features):
				best_features.append(self.columns[indices[f]])
			f += 1

		return best_features
