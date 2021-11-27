#################################
# Your name: Rony Kositsky
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals
import math


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        xs = np.random.uniform(0, 1, m)  # generates m points in [0,1], with a uniform distribution
        xs.sort()
        return np.column_stack((xs, self.get_labels(xs, m)))

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        empirical_errors, true_errors = [], []
        x = [m for m in (range(m_first, m_last + step, step))]

        for m in x:

            T_emp_err, T_true_err = [], []
            for i in range(T):
                samples = self.sample_from_D(m)
                xs, ys = samples[:, 0], samples[:, 1]
                emp_I, emp_err = intervals.find_best_interval(xs, ys, k)  # find best interval by ERM
                T_true_err.append(self.calc_true_error(emp_I))
                T_emp_err.append(emp_err / m)
            ave_true_err = sum(T_true_err) / T
            ave_emp_err = sum(T_emp_err) / T
            true_errors.append(ave_true_err)
            empirical_errors.append(ave_emp_err)

        plt.plot(x, empirical_errors, label="empirical errors")
        plt.plot(x, true_errors, label="true errors")
        plt.legend()
        plt.show()
        return [[empirical_errors[i], true_errors[i]] for i in range(len(empirical_errors))]

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        samples = self.sample_from_D(m)
        xs, ys = samples[:, 0], samples[:, 1]
        x = [i for i in range(k_first, k_last + step, step)]
        empirical_errors, true_errors = [], []
        min_err_cnt, best_k_ERM = m, None

        for k in x:

            emp_I, emp_err = intervals.find_best_interval(xs, ys, k)
            empirical_errors.append(emp_err / m)
            true_errors.append(self.calc_true_error(emp_I))

            if emp_err < min_err_cnt:
                best_k_ERM = k
                min_err_cnt = emp_err

        plt.plot(x, empirical_errors, label="empirical errors")
        plt.plot(x, true_errors, label="true errors")
        plt.legend()
        plt.show()

        return best_k_ERM

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        samples = self.sample_from_D(m)
        xs, ys = samples[:, 0], samples[:, 1]
        x = [i for i in range(k_first, k_last + step, step)]
        empirical_errors, true_errors, penalties, penalties_and_empirical_errors = [], [], [], []
        min_err_cnt, best_k_ERM = m, None

        for k in x:
            emp_I, emp_err = intervals.find_best_interval(xs, ys, k)

            empirical_errors.append(emp_err / m)
            true_errors.append(self.calc_true_error(emp_I))
            penalty = self.penalty_calculation(k, 0.1, m)
            penalties.append(penalty)
            penalties_and_empirical_errors.append(emp_err / m + penalty)

            if emp_err < min_err_cnt:
                best_k_ERM = k
                min_err_cnt = emp_err

        plt.plot(x, empirical_errors, label="empirical errors")
        plt.plot(x, true_errors, label="true errors")
        plt.plot(x, penalties, label="penalties")
        plt.plot(x, penalties_and_empirical_errors, label="penalty + empirical error")
        plt.legend()
        plt.show()
        return best_k_ERM

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.
        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        samples = self.sample_from_D(m)
        best_k = 0
        best_valid_error = 1

        train_sample = int(0.8 * m)
        xs_train = samples[:train_sample, 0]
        ys_train = samples[:train_sample, 1]
        xs_valid = samples[train_sample:, 0]
        ys_valid = samples[train_sample:, 1]

        for k in range(11):
            emp_I, emp_err = intervals.find_best_interval(xs_train, ys_train, k)
            emp_err_holdout = self.calc_emp_err(emp_I, xs_valid, ys_valid)
            if emp_err_holdout < best_valid_error:
                best_valid_error = emp_err_holdout
                best_k = k

        return best_k

    #################################

    @staticmethod
    def get_labels(xs, m):
        """returns labels for the points in xs, based on the given distribution"""
        labels = []
        for i in range(m):
            label = np.random.choice([0, 1], size=1, p=[0.2, 0.8]) if (
                        xs[i] <= 0.2 or (0.4 <= xs[i] <= 0.6) or (0.8 <= xs[i] <= 1.0)) \
                else np.random.choice([0, 1], size=1, p=[0.9, 0.1])
            labels.append(label[0])
        return labels

    def calc_true_error(self, interval):
        I_neg = []
        true_error = 0
        for i in range(len(interval) - 1):
            I_neg.append((interval[i][1], interval[i + 1][0]))
        true_true_intervals = self.get_overlap(interval, [(0, 0.2), (0.4, 0.6), (0.8, 1.0)])  # error = 0.2
        true_false_intervals = 0.6 - true_true_intervals  # error = 0.8
        false_false_intervals = self.get_overlap(I_neg, [(0.2, 0.4), (0.6, 0.8)])  # error = 0.1
        false_true_intervals = 0.4 - false_false_intervals  # error = 0.9
        true_error += true_true_intervals * 0.2 + true_false_intervals * 0.8 + false_false_intervals * 0.1 + false_true_intervals * 0.9
        return true_error

    def get_overlap(self, inter, real_interval):
        overlap = 0
        for i in range(len(real_interval)):
            start_true = real_interval[i][0]
            end_true = real_interval[i][1]
            for interval in inter:
                start_emp = interval[0]
                end_emp = interval[1]
                start_overlap = max(start_emp, start_true)
                end_overlap = min(end_emp, end_true)
                if start_overlap < end_overlap:
                    length = end_overlap - start_overlap
                    overlap += length
        return overlap

    @staticmethod
    def penalty_calculation(k, delta, n):
        return 2 * (((2 * k + math.log(2 / delta)) / n) ** 0.5)

    @staticmethod
    def calc_emp_err(interval, xs_valid, ys_valid):
        err_cnt = 0
        for i in range(len(xs_valid)):
            label = False
            for interval in interval:
                if interval[0] <= xs_valid[i] <= interval[1]:
                    label = True
                    break
            if (label and ys_valid[i] == 0) or (not label and ys_valid[i] == 1):
                err_cnt += 1
        return err_cnt / len(xs_valid)

    #################################


if __name__ == '__main__':
    ass = Assignment2()
    # ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    # ass.experiment_k_range_srm(1500, 1, 10, 1)
    # ass.cross_validation(1500)
