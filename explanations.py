"""
Module containing text explanations for the perceptron learning algorithm.
These are displayed to the user during the execution of the perceptron simulation.
"""

BASIC_EXPLANATION = """
A perceptron is a simple artificial neuron that takes binary inputs (0 or 1) and produces 
a binary output. It works by:

1. Multiplying each input by a weight
2. Summing these weighted inputs (plus an optional bias term)
3. Outputting 1 if this sum exceeds a threshold, otherwise 0

During learning, the perceptron adjusts its weights to correctly classify all training examples.
The weights are updated whenever the perceptron makes a mistake.

A perceptron can learn simple functions like AND and OR, but not XOR (which requires multiple layers).
"""

DETAILED_EXPLANATION = """
=======================================================================
                    PERCEPTRON LEARNING ALGORITHM
=======================================================================

WHAT IS A PERCEPTRON?
---------------------
A perceptron is a simple type of artificial neuron that takes multiple binary (0 or 1) 
inputs and produces a single binary output. It's one of the earliest machine learning
algorithms, designed to mimic how a single brain neuron works.

KEY COMPONENTS:
--------------
1. INPUTS:
   Binary values (0 or 1) that represent the features of our data.
   Example: In a logical AND function, we have two inputs (a and b).

2. WEIGHTS:
   Each input has an associated weight that determines its importance.
   These weights are adjusted during learning to make the perceptron produce the correct output.

3. BIAS:
   An additional parameter that allows the perceptron to shift its activation threshold.
   Without a bias, the perceptron's decision boundary would always pass through the origin.

4. ACTIVATION:
   The weighted sum of all inputs plus the bias.
   Activation = (input1 × weight1) + (input2 × weight2) + ... + bias

5. THRESHOLD:
   A value that the activation is compared against to determine the output.
   If activation ≥ threshold, output = 1; otherwise, output = 0.

HOW LEARNING WORKS:
------------------
1. INITIALIZATION:
   - Start with random or zero weights
   - Set an initial bias value
   - Define a learning rate (how quickly weights change)

2. FOR EACH TRAINING EXAMPLE:
   a) Calculate the activation and output based on current weights and bias
   b) Compare the output with the expected target value
   c) If they don't match, update the weights and bias:
      - Calculate error: delta = learning_rate × (target - output)
      - Update weights: new_weight = old_weight + (delta × input)
      - Update bias: new_bias = old_bias + delta

3. REPEAT UNTIL CONVERGENCE:
   Keep presenting all training examples and updating weights until no changes
   are needed, or until a maximum number of epochs is reached.

MATHEMATICAL FORMULAS:
---------------------
1. Activation formula:
   a = (x₁ × w₁) + (x₂ × w₂) + ... + (xₙ × wₙ) + b
   Where:
   - a is the activation
   - x₁, x₂, ..., xₙ are the input values
   - w₁, w₂, ..., wₙ are the weights
   - b is the bias

2. Output formula:
   output = {
     1  if  a ≥ threshold
     0  if  a < threshold
   }

3. Weight update formula:
   wᵢ_new = wᵢ_old + learning_rate × (target - output) × xᵢ

4. Bias update formula:
   b_new = b_old + learning_rate × (target - output)

THRESHOLD SELECTION:
------------------
The threshold determines when the perceptron outputs 1 vs 0. It can be:

1. Manually specified based on the logic function:
   - AND functions: threshold ≈ number of inputs (e.g., 1.5-2.0 for 2 inputs)
   - OR functions: threshold ≈ 0.5
   - NOT functions: threshold ≈ 0.5 with negative weights

2. Automatically calculated:
   - Find highest activation for target=0 inputs
   - Find lowest activation for target=1 inputs
   - Set threshold at midpoint between these values

This automatic calculation creates an optimal decision boundary for the 
current weights and bias values.

LIMITATIONS:
-----------
Perceptrons can only learn linearly separable functions. This means they can learn
simple functions like AND, OR, and NOT, but cannot learn XOR without additional layers.

=======================================================================

The following simulation will walk through the perceptron learning process step-by-step,
showing all calculations and weight updates in detail.
"""