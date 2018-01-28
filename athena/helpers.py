from __future__ import print_function

def print_statistics(testing_targets, predictions):
	"""
    Helper function for printing curve fitting statistics.

    Athena uses a random forest from Scikit-Learn to select the best parameters. Use this constructor to create a Selection class.

    Parameters
    ----------
    testing_targets : ?
        Number of estimators (decision trees) the Random Forest will use.
    predictions : ?
        Number of threads the Random Forest will utilize.
    """
	from sklearn.metrics import mean_squared_error
	from scipy.stats import pearsonr
	from numpy import percentile, abs, sqrt

	percentile_error = lambda y_true, y_pred, pcnt: percentile(abs(y_true - y_pred), pcnt)

	tp = testing_targets, predictions
	output = [
		pearsonr(*tp)[0] * 100,
		sqrt(mean_squared_error(*tp)),
		percentile_error(*tp, pcnt=90),
		percentile_error(*tp, pcnt=95),
		percentile_error(*tp, pcnt=99),
	]

	print('\t'.join([str(round(x)) for x in output]))


def split_dataframe(data_frame, split=0.9):
	"""
    Helper function for splitting data-set into training and testing sets.

    Athena uses a random forest from Scikit-Learn to select the best parameters. Use this constructor to create a Selection class.

    Parameters
    ----------
    data_frame : pandas.DataFrame
        Number of estimators (decision trees) the Random Forest will use.
    split : float
        Number of threads the Random Forest will utilize.
    """
	from numpy import random
	from pandas import DataFrame

	assert isinstance(data_frame, DataFrame)
	assert 0.0 < split < 1.0

	data_frame["is_training"] = random.rand(len(data_frame), 1) <= split
	training_df = data_frame[data_frame["is_training"] == True]
	testing_df = data_frame[data_frame["is_training"] == False]
	data_frame.drop("is_training", axis=1, inplace=True)

	return training_df, testing_df


def mean_confidence_interval (a, confidence=0.95):
	"""
    Helper function for calculating a mean confidence interval

    Athena uses a random forest from Scikit-Learn to select the best parameters. Use this constructor to create a Selection class.

    Parameters
    ----------
    a : ?
        Number of estimators (decision trees) the Random Forest will use.
    confidence : float
        Number of threads the Random Forest will utilize.
    """
	from numpy import mean
	from scipy.stats import sem, t
	n = len(a)
	m, se = mean(a), sem(a)
	h = se * t._ppf((1 + confidence) / 2., n - 1)
	return m, m - h, m + h
