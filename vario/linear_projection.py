import numpy as np

from vario.linear_regression import linear_regression
from vario.utils import logg
from vario.utils_context_partition import partition_to_groupmap


class LinearProjection():
    """
    Given data in n contexts and a target variable Y with covariates X,
    we represent this data by the linear parameters of f:X -> Y in each context
    """

    def __init__(self, data_each_context, index_Y, indices_X=None):
        self.index_Y = index_Y
        self.indices_X = indices_X

        all_nodes = data_each_context[0].shape[1]
        if indices_X is None:
            self.n_nodes = all_nodes
        else:
            self.n_nodes = len(indices_X)
        assert 0 < self.n_nodes <= all_nodes

        self.n_contexts = len(data_each_context)
        self.contexts = range(self.n_contexts)
        self.nodes = range(self.n_nodes)
        assert 0<=index_Y<all_nodes

        self.data_X = np.array([data_each_context[c_i][:, [i for i in self.indices_X]] for c_i in range(self.n_contexts)])
        self.data_Y = np.array([data_each_context[c_i][:, self.index_Y] for c_i in range(self.n_contexts)])

        self.linear_regression()


    def linear_regression(self):
        """Linear regression per context

        """
        self.parameters = linear_regression(self.data_X, self.data_Y, scale=True) #column j: coefficients for variable j in each context

    def score(self, partition,
              empirical=True,
              resid=False):
        """ Vario score

        :param pi_test: partition
        :param empirical: empirical score on euclidean distance (typically better for finding mechanism changes)
        :param resid: full MDL score including residuals (not so good unless the true data generating process is linear)
        :return: score
        """
        map = partition_to_groupmap(partition, self.n_contexts)

        param_group = [[sum([self.parameters[:, x_i][c_i] for c_i in partition[pi_k]]) / len(partition[pi_k])
                        for pi_k in range(len(partition))]
                       for x_i in self.nodes] #[[c(x1, pi1),...],#[c(x2, pi1),...]]

        param_error = [[(self.parameters[:, x_i][c_i] - param_group[x_i][map[c_i]]) ** 2
                        for c_i in self.contexts]
                      for x_i in self.nodes]
        param_onegroup = [[sum(self.parameters[:, x_i]) / self.n_contexts]
                          for x_i in self.nodes]
        param_error_onegroup = [[(self.parameters[:, x_i][c_i] - param_onegroup[x_i][0]) ** 2 for c_i in self.contexts]
                               for x_i in self.nodes]


        if empirical:
            sse = [logg(sum(param_error[x_i]))
                    for x_i in self.nodes]
            sse_one = [logg(sum(param_error_onegroup[x_i]))
                    for x_i in self.nodes]
            vario_score = sum([(sse_one[x_i] - sse[x_i]) / max(len(partition) - 1, 1)
                               for x_i in self.nodes])
        else:
            sig = 1
            mdl_coef = sum([(self.n_contexts / 2) * logg(2 * np.pi * (sig ** 2)) + 1 / (2 * (sig ** 2)) * sum(param_error[x_i])
                            for x_i in self.nodes]) # coefficient errors are per dimension
            mdl_alpha = (logg(self.n_contexts) / 2) * len(partition)
            mdl_pi = self.n_contexts * logg(self.n_contexts) + logg(self.n_contexts) # TODO constant model cost aspects?

            if resid:
                num_samples = sum([len(self.data_Y[c_i]) for c_i in self.contexts])

                residuals_context = [sum((sum([self.parameters[c_i, x_i]*self.data_X[c_i][:,x_i] for x_i in self.nodes])
                                   - self.data_Y[c_i]) ** 2) for c_i in self.contexts]

                mdl_data = (num_samples / 2) * logg(2 * np.pi * (sig ** 2)) \
                           + 1 / (2 * (sig ** 2)) * (logg(sum(residuals_context)))
                vario_score = mdl_coef + mdl_alpha + mdl_pi + mdl_data

            else:
                vario_score = mdl_coef + mdl_alpha + mdl_pi
        return vario_score
