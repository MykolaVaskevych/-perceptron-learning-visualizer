"""
A non-interactive test run of the perceptron learning algorithm.
This script runs without requiring user input.
"""

from perceptron import perceptron_learning
from visualization import print_verification_modular, print_final_model_explanation_modular
from utils import parse_logic_function, suggest_initial_parameters

def test_and_function():
    """Test the perceptron on the AND function."""
    # Configuration
    logic_function = "a and b"
    num_vars, _ = parse_logic_function(logic_function)
    
    # Get suggested parameters
    suggested_threshold, suggested_bias, suggested_skip, _ = suggest_initial_parameters(
        logic_function, num_vars
    )
    
    print(f"Testing perceptron on: {logic_function}")
    print(f"Using threshold: {suggested_threshold}, bias: {suggested_bias}")
    
    # Run perceptron learning
    final_weights, epochs, _, final_bias = perceptron_learning(
        logic_function,
        num_vars,
        learning_rate=1.0,
        threshold=suggested_threshold,
        bias_init=suggested_bias,
        weights_init=None,
        max_epochs=100,
        skip_zeros=suggested_skip,
        show_details=False  # Don't show detailed training process
    )
    
    # Format weights for display
    weights_display = ", ".join([f"{w:.2f}" for w in final_weights])
    
    # Display results
    print("\nTRAINING RESULTS:")
    print(f"Epochs to converge: {epochs}")
    print(f"Final weights: [{weights_display}]")
    print(f"Final bias: {final_bias:.2f}")
    
    # Verify the model
    accuracy = print_verification_modular(
        final_weights, final_bias, logic_function, num_vars, suggested_threshold, "basic"
    )
    
    # Print final model explanation
    print_final_model_explanation_modular(
        final_weights, final_bias, suggested_threshold, num_vars, accuracy, logic_function, "basic"
    )

if __name__ == "__main__":
    test_and_function()