# Import necessary libraries
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import pandas as pd

# Load data
genetic_data = pd.read_csv('../data/genetic_data.csv')
geographic_data = pd.read_csv('../data/geographic_data.csv')

# Combine data for simplicity in this example (tailor to your use case)
data = pd.merge(genetic_data, geographic_data, on="patient_id")

# Define Bayesian Network structure based on the conditional dependencies
model = BayesianNetwork([
    ('GeneticFactor', 'Disease'),
    ('GeographicLocation', 'Disease'),
    ('Age', 'Disease')
])

# Fit model using Maximum Likelihood Estimation
model.fit(data, estimator=MaximumLikelihoodEstimator)

# Save the model structure for reference
model.to_bif("disease_diagnosis.bif")

# Perform inference to make a diagnosis
inference = VariableElimination(model)
result = inference.map_query(variables=['Disease'], evidence={'GeneticFactor': 'present', 'GeographicLocation': 'rural'})
print("Diagnosis:", result)
