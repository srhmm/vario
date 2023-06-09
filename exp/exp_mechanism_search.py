import numpy as np

from vario.causal_mechanism_search import causal_mechanism_search
from vario.generate_data import generate_data_from_DAG
from vario.utils_confidence import confident_score, conf_intervals_gaussian
from vario.utils_context_partition import enum_context_partitions, pi_matchto_pi_pairwise, pi_matchto_pi_exact


def test_mechanism_search(n_reps=100, test_greedy_version=False):
    print(f"Evaluating VARIO_PI: Discovery of Causal Mechanism Changes (reps={n_reps})")
    greedy = test_greedy_version
    n_nodes = 5
    n_contexts = 5
    n_samples = 1000
    partitions = enum_context_partitions(n_contexts, permute=False)
    conf_intervals, _ = conf_intervals_gaussian(n_contexts)

    # Partition Discovery
    exact_matches, no_matches = [0 for _ in range(n_contexts)], [0 for _ in range(n_contexts)]
    TP, FP, TN, FN = [0 for _ in range(n_contexts)], [0 for _ in range(n_contexts)],[0 for _ in range(n_contexts)], [0 for _ in range(n_contexts)]

    for true_partition in partitions:
        for seed in range(n_reps):

            # Generating data
            data, index_Y, true_dag, true_partitions = generate_data_from_DAG(n_nodes, n_contexts, n_samples,
                                                                              partition_Y=true_partition,
                                                                              min_interventions=5, max_interventions=5,
                                                                              scale=False, verbose=False, seed=seed)


            indices_X = true_dag.parents_of(index_Y)
            estim_partition, estim_score, _ = causal_mechanism_search(data, index_Y, indices_X, greedy=greedy,
                                                                      verbose=False)


            is_significant = confident_score(conf_intervals, estim_partition, estim_score, n_contexts)

            # insignificant score: decide on no groups
            if not is_significant:
                estim_partition = [[c_i for c_i in range(n_contexts)]]

            # special case: one group is the same as no groups
            if len(estim_partition)==n_contexts and len(true_partition) == 1 or \
                len(true_partition) == n_contexts and len(estim_partition) == 1:
                estim_partition = true_partition

            # Consider exact match
            match, nomatch = pi_matchto_pi_exact(estim_partition, true_partition)

            # Consider clustering accuracy: is a pair of contexts assigned to the same/a different group correctly?
            tp, fp, fn, tn, _, _ = pi_matchto_pi_pairwise(true_partition, estim_partition, n_contexts)

            k = (len(true_partition)-1)
            exact_matches[k] = exact_matches[k] + match
            no_matches[k] = no_matches[k] + nomatch
            TP[k], FP[k] = TP[k] + tp, FP[k] + fp
            TN[k], FN[k] = TN[k] + tn, FN[k] + fn


    print ("\nAccuracy on exactly discovering context partitions\nwith K groups in the context partition/K-1 causal mechanism changes")
    for k in range(n_contexts-1):
        print(f"\tK={k+1}: {np.round(exact_matches[k]/max(exact_matches[k]+no_matches[k],1),2)} ({exact_matches[k]}/{exact_matches[k]+no_matches[k]})")

    print("\nF1 (TP, TN, FP, FN) on correctly assigning pairs of contexts to the same/a different group\nwith K groups in the context partition/K-1 causal mechanism changes")
    for k in range(n_contexts-1):
        print(f"\tK={k+1}: {np.round(TP[k]/max(TP[k]+1/2*(FP[k]+FN[k]),1),2)} ({TP[k]},{TN[k]},{FP[k]},{FN[k]})")

    #If there are no groups, F1 score doesn't make much sense, as there are no TPs (and we want TNs)
    k = n_contexts-1
    print(f"\tK={k}: -- ({TP[k]},{TN[k]},{FP[k]},{FN[k]})")
