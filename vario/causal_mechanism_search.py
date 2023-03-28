import numpy as np
from typing import List

from vario.partition_record import PartitionRecord
from vario.utils_context_partition import enum_context_partitions, split_partition



def causal_mechanism_search(data_each_context: List,
                            iy: int, ix: List, greedy=True, verbose=False,
                            partition_record=None):
    """ Discovers a partition of n contexts into k groups.
    If contexts i,j are in the same group, the functions fi: X -> Y and fj : X -> Y in these contexts are similar.

    :param data_each_context: list
    :param iy: which data row is target/child Y
    :param ix: which data rows are covariates/causal parents X
    :param greedy: whether to traverse all possible partitions of the contexts or greedy top-down variant (cubic time)
    :param verbose: printing
    :return: partition, score, PartitionRecord
    """
    if partition_record is None:
        partition_record = PartitionRecord(data_each_context)
    if greedy:
        return greedy_mechanism_search(partition_record, iy, ix, verbose)
    return exhaustive_mechanism_search(partition_record, iy, ix, verbose)


def greedy_mechanism_search(pi_record: PartitionRecord, iy: int, ix: List, verbose):
    """ Top down tree search for a partition of n contexts.

    :param pi_record: PartitionRecord
    :param iy: target/child Y
    :param ix: covariates/causal parents X
    :param verbose: printing
    :return: best partition, best score, updated pi_record
    """

    # Regression over X and Y in all contexts
    linmap = pi_record.eval_regression_function(iy, ix)

    best_score, best_pi = -np.inf, None

    # Start with k=1 and the one-group partition (no mechanism change)
    Piall = [[c_i for c_i in linmap.contexts]]
    best_score_atk = pi_record.eval(iy, ix, Piall)
    best_pi_atk = Piall

    if verbose:
        print("\n--- Top Down Tree Search for Partitions ---")
        print("Target: Y =", iy)
        print("\t1 Group:", round(best_score_atk, 2), best_pi_atk)

    # Increase the number of mechanism changes
    for k in range(2, linmap.n_contexts + 1):

        # Consider all ways to split the best partition at the previous level of the search tree
        candidates = split_partition(best_pi_atk)

        # Among the candidates, find the best partition with k groups/k-1 mechanism changes
        best_score_atk, best_pi_atk = -np.inf, None
        for partition in candidates:
            score = pi_record.eval(iy, ix, partition)
            if score > best_score_atk:
                best_score_atk, best_pi_atk = score, partition
            if score > best_score:
                best_score, best_pi = score, partition
        if verbose:
            print("\t" + str(k), "Groups:", round(best_score_atk, 2), best_pi_atk)

    if verbose:
        print("\tOverall:", round(best_score, 2), best_pi)

    # for (score, pi) in pi_record.sorted_partitions(iy, ix):
    #     print(score, pi)
    return best_pi, best_score, pi_record


def exhaustive_mechanism_search(pi_record: PartitionRecord, iy: int, ix: List, verbose):
    """ Exhaustive search for a partition of n contexts.

    :param pi_record: PartitionRecord
    :param iy: target/child Y
    :param ix: covariates/causal parents X
    :param verbose: printing
    :return: best partition, updated pi_record
    """

    # Regression over X and Y in all contexts
    linmap = pi_record.eval_regression_function(iy, ix)

    Pizero = [[c_i] for c_i in linmap.contexts]
    best_score, best_pi = -np.inf, Pizero

    if verbose:
        print("\n--- Exhaustive Search over Partitions ---")
        print("Target: Y =", iy)
    context_partitions = enum_context_partitions(linmap.n_contexts)
    for partition in context_partitions:

        score = pi_record.eval(iy, ix, partition)

        if score > best_score:
            best_score, best_pi = score, partition

    if best_score == -np.Inf:
        raise RuntimeError("No min found")

    if verbose:
        print("Best Partition:", best_pi, "Score:", round(best_score,2))

    # for (score, pi) in pi_record.sorted_partitions(iy, ix):
    #     print(score, pi)
    return best_pi, best_score, pi_record
