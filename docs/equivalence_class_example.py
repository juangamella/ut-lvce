import utlvce
import utlvce.generators as generators
import utlvce.utils as utils

# Generate a random model and sample some data

model = generators.random_graph_model(
    20, 2.1, {1, 2, 3}, 2, 5, 0.5, 0.6, 3, 6, 0.2, 0.3, 0.2, 1, 0.7, 0.8)
data = model.sample(1000, compute_sample_covariances=False, random_state=42)

# Find the equivalence class of the data generating model using the true MEC as candidate graphs
candidate_dags = utils.mec(model.A)
estimated_icpdag, estimated_I, estimated_model = utlvce.equivalence_class(
    candidate_dags, data, prune_edges=False, verbose=1)

# View results
print(estimated_icpdag)
print(estimated_I)
print(estimated_model)
