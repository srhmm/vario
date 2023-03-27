### Discovering Invariant and Changing Mechanisms from Data

#### Vario Algorithms
- `VARIO_PI`: Given a variable and its causes in multiple contexts, discovers a partition of the contexts, showing where the causal mechanism is invariant and where it changes. See `causal_mechanism_search.py`

- `VARIO`: Given a variable observed in multiple contexts, discovers a context partition and the causal parents of that variable. See `causal_variable_search.py`.
- `VARIO_G`: For a set of variables observed in multiple contexts, discovers context partitions and causal parents of each variable. See `causal_dag_search.py`.


#### Quick Example
```
data, index_Y, true_dag, true_partitions = generate_data_from_DAG(n_nodes=5, n_contexts=5, n_samples_per_context=500,
                                                                partition_Y=[[0, 1, 2],[3,4]],
                                                                min_interventions=5,max_interventions=5, scale=False, verbose=True, seed=3) 
                                                                
# VARIO_PI
oracle_partition, oracle_score, _ = causal_mechanism_search(data, index_Y, true_dag.parents_of(index_Y), greedy=False, verbose=True)

# VARIO
estim_parents, estim_partition, estim_score, _ = causal_variable_search(data, index_Y,  [n for n in range(n_nodes) if not (n==index_Y)], greedy_mechanism_search=False, verbose=True)

```