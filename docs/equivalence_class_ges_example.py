import utlvce
import utlvce.generators as generators
import utlvce.utils as utils

# Generate a random model and sample some data
targets = generators.intervention_targets(10, 5)
model = generators.random_graph_model(
    10, 1.5, targets, 1, 5, 0.5, 0.6, 3, 6, 0.2, 0.3, 0.2, 1, 0.7, 0.8)
data = model.sample(1000, random_state=42)

# Find the equivalence class of the data generating model using GES to obtain the initial set of candidate graphs
candidate_dags = utils.mec(model.A)
estimated_icpdag, estimated_I, estimated_model = utlvce.equivalence_class_w_ges(
    data, verbose=1, nums_latent=[2])

# View results
print(estimated_icpdag)
print(estimated_I)
print(estimated_model)
