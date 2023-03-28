import numpy as np
from sklearn import preprocessing

from vario.utils import cantor_pairing


def scale_data(y):
    scaler = preprocessing.StandardScaler().fit(y)
    return(scaler.transform(y))


def data_groupby_pi(Xc, yc, Pi):
    """Given a partition Pi={pi_k}, pool data within each group pi_k"""
    n_pi = len(Pi)
    n_c = Xc.shape[0]

    assert((len(Xc.shape) == 3) & (len(yc.shape) == 2))
    assert((yc.shape[0] == n_c) & (yc.shape[1] == Xc.shape[1]))

    Xpi = [np.concatenate([Xc[c_i] for c_i in Pi[pi_k]]) for pi_k in range(n_pi)]
    ypi = [np.concatenate([yc[c_i] for c_i in Pi[pi_k]]) for pi_k in range(n_pi)]

    return(Xpi, ypi)


def gen_arc_weights_pi(partition, dag, seed, const_intervention=False):

    arc_weights_k = [{} for _ in range(len(partition))]
    if len(partition) == 1:
        for (i, j) in dag.arcs:
            arc_weights_k[0][(i, j)] = dag.arc_weights[(i, j)]
        return arc_weights_k

    #mn = 80 / max(len(partition) - 1, 1)  # weights at least in [0,80]
    #mx = 100 / max(len(partition) - 1, 1)

    for (i, j) in dag.arcs:
        seed_ij = cantor_pairing(seed, cantor_pairing(i, j))
        for k in range(len(partition)):
            if k % 2 == 0:
                sign = 1
                off = k/2
            else:
                sign = -1
                off = -(k-1)/2
            arc_weights_k[k][(i, j)] =  np.random.RandomState(seed_ij).normal(loc=sign * dag.arc_weights[(i, j)] + off, scale=.5)

        # special case: if iv_type is hard/surgical intervention: set  arc weights in one group to zero
        if const_intervention:
            arc_weights_k[0][(i, j)] = 0
    return arc_weights_k

