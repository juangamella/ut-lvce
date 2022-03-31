import utlvce
import utlvce.generators as generators
import utlvce.utils as utils

# Generate a random model and sample some data
model = generators.random_graph_model(
    20, 2.1, {1, 2, 3}, 2, 5, 0.5, 0.6, 3, 6, 0.2, 0.3, 0.2, 1, 0.7, 0.8)
data = model.sample(1000, random_state=42)

# For algorithm 1 and 2, we will use the Markov Equivalence class of the data generating
# graph as a candidate set
candidate_dags = utils.mec(model.A)

# Algorithm 1: Estimate equivalence class of the best scoring
# candidate
utlvce.equivalence_class(candidate_dags, data, prune_edges=False)

# Algorithm 2: Improve on the initial candidate set and estimate the
# equivalence class of the best scoring graph
utlvce.equivalence_class(candidate_dags, data, prune_edges=True)

# GES + UT-LVCE: Estimate equivalence class of the data generating
# model using GES to obtain the initial set of candidate graphs
utlvce.equivalence_class_w_ges(data)
