from pandas import DataFrame
from athena.equations import Equation

class Dataset():
	def __init__(self, training_df: DataFrame, testing_df: DataFrame, parameters_map=None):
		if parameters_map is None:
			parameters_map = {"not_normalized": {}}
			for l in list(training_df):
				parameters_map["not_normalized"][l] = l

		assert isinstance(training_df, DataFrame)
		assert isinstance(testing_df, DataFrame)
		assert isinstance(parameters_map, dict)

		self.ds = {"training": {}, "testing": {}}

		if "normalized" in parameters_map:
			for column_name, variable_name in parameters_map["normalized"].items():
				assert isinstance(column_name, str)
				assert isinstance(variable_name, str)
				assert column_name in list(training_df)
				assert column_name in list(testing_df)

				# ==========================================================================================
				# Parameters here require normalization; testing data-set uses
				# normalization min/max from training for consistency.
				# ==========================================================================================
				self.ds["training"][column_name] = Equation.normalize(training_df[column_name].values, variable_name)
				self.ds["testing"][column_name] = Equation.normalize(testing_df[column_name].values, variable_name,
													   min=self.ds["training"][column_name][-1][0],
													   max=self.ds["training"][column_name][-1][1])

				self.ds["training"][column_name] = self.ds["training"][column_name][0:2]
				self.ds["testing"][column_name] = self.ds["testing"][column_name][0:2]

		if "not_normalized" in parameters_map:
			for column_name, variable_name in parameters_map["not_normalized"].items():
				assert isinstance(column_name, str)
				assert isinstance(variable_name, str)
				assert column_name in list(training_df)
				assert column_name in list(testing_df)

				# ==========================================================================================
				# Parameters here require no normalization at all.
				# ==========================================================================================
				self.ds["training"][column_name] = training_df[column_name].values, variable_name
				self.ds["testing"][column_name] = testing_df[column_name].values, variable_name

	def get(self, p: str):
		assert p in self.ds
		return self.ds[p]

