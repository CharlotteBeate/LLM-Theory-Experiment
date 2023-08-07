###############################################
# This script tests whether the temperature-scaled softmax probabilities are close to the transformed probabilities.
# The temperature-scaled softmax probabilities are computed using the temperature-scaled softmax function.
# this is to confirm that my math is corrct
import numpy as np

def softmax(logits):
    logits_exp = np.exp(logits)
    return logits_exp / np.sum(logits_exp)

def temperature_scaled_softmax(logits, T):
    scaled_logits = logits / T
    scaled_logits_exp = np.exp(scaled_logits)
    return scaled_logits_exp / np.sum(scaled_logits_exp)

def transformation(P, T):
    P_transformed = np.power(P, 1/T)
    return P_transformed / np.sum(P_transformed)

n= 5

def test_temperature_scaled_softmax(n):
    # Generate random logits and temperature
    logits = np.random.rand(n)
    T = np.random.uniform(0, 2)

    # Compute softmax and temperature-scaled softmax of logits
    P = softmax(logits)
    P_T = temperature_scaled_softmax(logits, T)

    # Apply transformation to softmax probabilities
    P_transformed = transformation(P, T)

    # Check that temperature-scaled softmax is close to transformed probabilities
    epsilon = 1e-8
    if not np.allclose(P_T, P_transformed, atol=epsilon, rtol=epsilon):
        print("Error: Temperature-scaled softmax probabilities and transformed probabilities are not close.")
        print(f"Temperature: {T}")
        print(f"Logits: {logits}")
    # else:
    #     print("Success: Temperature-scaled softmax probabilities and transformed probabilities are close.")

# Run test 200 times
for _ in range(200):
    test_temperature_scaled_softmax(n)
