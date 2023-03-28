import numpy as np

from vario.causal_mechanism_search import causal_mechanism_search
from vario.causal_variable_search import causal_variable_search
from vario.generate_data import generate_data_from_DAG
from vario.utils_confidence import confident_score, conf_intervals_gaussian
from vario.utils_context_partition import pi_matchto_pi_exact, enum_context_partitions, pi_matchto_pi_pairwise
from vario.utils_eval import eval_causal_edge


def test_causal_variable_search(n_reps=100, test_greedy_version=False, verbose=False):
    print(f"Evaluating VARIO: Discovery of Causal Mechanism Changes and Causal Parents (reps={n_reps})")
    greedy = test_greedy_version
    n_nodes = 5
    n_contexts = 5
    n_samples = 1000
    partitions = enum_context_partitions(n_contexts, permute=True, k_min=2, k_max=2)
    conf_intervals, _ = conf_intervals_gaussian(n_contexts)

    exact_matches, no_matches = [0 for _ in range(n_contexts)], [0 for _ in range(n_contexts)]
    TN, FN = [0 for _ in range(n_contexts)], [0 for _ in range(n_contexts)]
    TP, FP = [0 for _ in range(n_contexts)], [0 for _ in range(n_contexts)]
    TPca, FPca, FPrev = [0 for _ in range(n_contexts)], [0 for _ in range(n_contexts)], [0 for _ in range(n_contexts)]
    TNca, FNca = [0 for _ in range(n_contexts)], [0 for _ in range(n_contexts)]

    for true_partition in partitions:
        for seed in range(n_reps):

            # Generating data
            data, index_Y, true_dag, true_partitions = generate_data_from_DAG(n_nodes, n_contexts, n_samples,
                                                                              partition_Y=true_partition,
                                                                              min_interventions=5, max_interventions=5,
                                                                              scale=False, verbose=False, seed=seed)
            indices_X = [n for n in range(n_nodes) if not (n == index_Y)]
            estim_parents, estim_partition, estim_score, _ = causal_variable_search(data, index_Y, indices_X,
                                                                              greedy_mechanism_search=greedy,
                                                                              verbose=verbose)
            if verbose:
                print("True parents, partition:\n{", ','.join([str(x) for x in true_dag.parents_of(index_Y)]), "}", true_partition, "\n")

            is_significant = confident_score(conf_intervals, estim_partition, estim_score, n_contexts)

            # insignificant score: decide on no groups
            if not is_significant:
                estim_partition = [[c_i for c_i in range(n_contexts)]]
                estim_parents = []

            # special case: one group is the same as no groups
            if len(estim_partition) == n_contexts and len(true_partition) == 1 or \
                    len(true_partition) == n_contexts and len(estim_partition) == 1:
                estim_partition = true_partition

            # Consider exact match
            match, nomatch = pi_matchto_pi_exact(estim_partition, true_partition)

            # Consider clustering accuracy: is a pair of contexts assigned to the same/a different group correctly?
            tp, fp, fn, tn, _, _ = pi_matchto_pi_pairwise(true_partition, estim_partition, n_contexts)

            k = (len(true_partition) - 1)
            exact_matches[k] = exact_matches[k] + match
            no_matches[k] = no_matches[k] + nomatch
            TP[k], FP[k] = TP[k] + tp, FP[k] + fp
            TN[k], FN[k] = TN[k] + tn, FN[k] + fn

            tpca, fpca, fprev, fnca, tnca =  eval_causal_edge(estim_parents, index_Y, true_dag)

            TPca[k], FPca[k], FPrev[k] = TPca[k] + tpca, FPca[k] + fpca, FPrev[k] + fprev
            TNca[k], FNca[k] = TNca[k] + tnca, FNca[k] + fnca

    print(
        "\nF1 (TP, TN, FP, FN) on correctly assigning pairs of contexts to the same/a different group")
    k=1
    print(
        f"\t{np.round(TP[k] / max(TP[k] + 1 / 2 * (FP[k] + FN[k]),1), 2)} ({TP[k]},{TN[k]},{FP[k]},{FN[k]})")


    print(
        "\nF1 (TP, TN, FP, FN) on correctly discovering causal variables")
    print(
        f"\t{np.round(TPca[k] / max(TPca[k] + 1 / 2 * (FPca[k] + FNca[k]),1), 2)} ({TPca[k]},{TNca[k]},{FPca[k]},{FNca[k]})")
    print(
        f"\nCausal Edges: Correctly Oriented ({TPca[k]}) vs. Missed ({FNca[k]}) vs. Oriented in the wrong direction ({FPrev[k]}) ")
