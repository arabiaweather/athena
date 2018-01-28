from __future__ import print_function
from athena.equations import *
import numpy as np
from athena.framework import Framework
from sympy.core.cache import clear_cache


class RandomSearch:
    def __init__(self, framework, search_length=100, equation_length=25, equations=None, starting_equations=None):
        """
        RandomSearch constructor function

        Parameters
        ----------
        framework : athena.Framework
        search_length : int
        equation_length : int
        equations : list
        starting_equations : list
        """
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
            # Below are the default starting equations; these may change in the future depending on experimental results
            self.functions = [SimpleSinusoidal, SimplePolynomial, BipolarPolynomial, Sinusoidal, MultiPolynomial]

    def iteration(self, return_constituents=False):
        """
        RandomSearch class's iteration function

        Parameters
        ----------
        return_constituents : bool
        """
        from random import choice, randint
        from sklearn.metrics import mean_absolute_error, r2_score
        from scipy.stats import pearsonr
        from copy import copy

        fw = copy(self.framework).reset()

        training_targets = fw.dataset.training_targets
        testing_targets = fw.dataset.testing_targets

        from athena.model import AdditiveModel
        model = AdditiveModel(fw)

        j = 0
        if self.starting_equations is not None:
            for se in self.starting_equations:
                try:
                    if choice([False, True]):
                        model.add(*se["args"], **se["kwargs"])
                        j += 1
                except:
                    continue

        model.add(Bias)

        # TODO: make sure we aren't wasting our time here by searching for an equation that has already been tried out before

        while j < self.equation_length:
            try:
                chosen_columns = []
                for i in range(randint(1, len(fw.dataset.get_columns()))):
                    chosen_columns.append(choice(fw.dataset.get_columns()))

                if choice([False, True]):
                    model.add(choice(self.functions), choice(self.functions), *chosen_columns)
                else:
                    model.add(choice(self.functions), *chosen_columns)

                j += 1
            except:
                continue

        self.searching_semaphore.acquire()

        try:
            fw.initialize(model, training_targets, loss_function="r2")
            fw.get_testing_predictions()

            # We will make this iteration stop earlier than planned if it can
            # be proven that this equation is diverging (going to infinity)!
            for fw_iteration in range(int(fw.max_iters)):
                fw.run_learning_step()
                if fw_iteration == int(int(fw.max_iters) / 10.0):
                    fw_predictions = fw.get_training_predictions()
                    if not np.isfinite(fw_predictions).all():
                        raise ValueError()

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

            if train_correlation > 0.25 and test_correlation > 0.25:
                error = {
                    "training_mae": training_error,
                    "testing_mae": testing_error,

                    "training_pearson": train_correlation,
                    "testing_pearson": test_correlation,

                    "training_r2": training_r2,
                    "testing_r2": testing_r2,
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

    def search(self, metric="testing_r2", return_constituents=False):
        """
        RandomSearch class's search function

        Parameters
        ----------
        metric : str
        return_constituents : bool
        """
        from tqdm import tqdm
        from threading import Thread, Semaphore
        from os import cpu_count
        import sys

        self.equations = []
        self.searching_threads = []
        self.searching_semaphore = Semaphore(value=cpu_count())

        for iteration in range(self.search_length):
            self.searching_threads.append(
                Thread(target=self.iteration, kwargs={"return_constituents": return_constituents}))
            self.searching_threads[-1].start()

        # TODO: find a better way to show progress feedback to the user
        # the problem with using tqdm on joining threads is that the
        # shown % done and iterations/second is inaccurate
        for thread in tqdm(self.searching_threads):
            thread.join()

        sys.stdout.flush()

        self.equations = sorted(self.equations, key=lambda x: x[metric])
        if "pearson" in metric or "r2" in metric:
            self.equations = self.equations[::-1]

    def get_best_equations(self, k=1):
        if k != 0:
            return self.equations[:k]
        else:
            return self.equations


class GeneticSearch:
    def __init__(self, fw, search_length, equation_length):
        """
        GeneticSearch constructor

        Parameters
        ----------
        fw : athena.Framework
        search_length : int
        equation_length: int
        """
        from math import inf as infinity
        self.fw = fw

        self.search_length = search_length
        self.equation_length = equation_length

        self.starting_equations = None
        self.iterations = 0

        self.best_r2_score = -1 * infinity

    def _remove_indices(self, x, indices):
        """
        GeneticSearch class private function to remove indices

        Parameters
        ----------
        x : ?
        indices : ?
        """
        return [i for j, i in enumerate(x) if j not in indices]

    def iteration(self):
        """
        GeneticSearch class's iteration function
        """
        self.iterations += 1

        while True:
            # We have to clear Sympy's cache before and after a random search,
            # otherwise we will bump into a Sympy cache key error.
            # The reason for this is unknown to me, but clearing the cache
            # has somehow solved this problem completely.
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
            elif equation[0]["testing_r2"] < self.best_r2_score:
                print("Iteration got {} which is less than {}, retrying.".format(equation[0]["testing_r2"], self.best_r2_score))
            else:
                print("Iteration produced {} equations.".format(len(equation)))
                equation = equation[0]
                self.best_r2_score = equation["testing_r2"]
                break

        self.starting_equations = equation["reconstruction"]
        return equation
