class Selection:
	def __init__(self, n_estimators=250, n_jobs=-1):
		from sklearn.ensemble import ExtraTreesRegressor
		self.clf = ExtraTreesRegressor(n_estimators=n_estimators, n_jobs=n_jobs)

	def train(self, data_frame, columns, target_column_name):
		# ==========================================================================================
		# Athena uses a random forest from Scikit-Learn to select the best parameters.
		# ==========================================================================================
		from pandas import DataFrame

		assert isinstance(data_frame, DataFrame)
		assert isinstance(columns, list)
		assert isinstance(target_column_name, str)

		self.columns = columns
		self.clf.fit(data_frame[columns].values, data_frame[target_column_name].values)
	
	def select(self, n, fixed_features=None):
		# ==========================================================================================
		# The best features array will be filled with non-fixed features.
		# ==========================================================================================
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
