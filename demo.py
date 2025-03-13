"""
Demonstration of the perceptron learning algorithm on various logic functions.
This script runs automatically without requiring user input.
"""

import numpy as np
from perceptron import perceptron_learning, simple_perceptron_learning
from visualization import (
    print_verification_modular,
    print_final_model_explanation_modular
)
from utils import parse_logic_function, suggest_initial_parameters


def run_demo(logic_function, use_simple_mode=False, detail_level="basic", max_epochs=100):
    """Run a demonstration of the perceptron learning algorithm on a given logic function."""
    print("\n" + "=" * 70)
    print(f"PERCEPTRON LEARNING DEMO: {logic_function}")
    print("=" * 70)

    # Parse the function to determine the number of variables
    try:
        num_vars, variables = parse_logic_function(logic_function)
        print(f"Detected variables: {', '.join(variables)} (total: {num_vars})")
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Get initial parameters
    suggested_threshold, suggested_bias, suggested_skip, _ = suggest_initial_parameters(
        logic_function, num_vars
    )

    print(f"Using suggested parameters:")
    print(f"- Threshold: {suggested_threshold}")
    if not use_simple_mode:
        print(f"- Initial bias: {suggested_bias}")
        print(f"- Skip zeros input: {'Yes' if suggested_skip else 'No'}")

    if use_simple_mode:
        # Simple mode configuration (no learning rate or bias)
        print("\nRunning in SIMPLE MODE (no learning rate or bias)")
        
        # Run simple perceptron learning
        final_weights, epochs, history = simple_perceptron_learning(
            logic_function,
            num_vars,
            suggested_threshold,
            None,  # weights_init
            max_epochs,
            detail_level == "detailed"
        )

        # Format weights for display
        weights_display = ", ".join([f"{w:.2f}" for w in final_weights])

        print("\n" + "=" * 50)
        print("TRAINING SUMMARY")
        print("=" * 50)
        print(f"Training completed in {epochs} epochs")
        print(f"Final weights: [{weights_display}]")

        # Verify the model (no bias in simple mode)
        accuracy = print_verification_modular(
            final_weights, 0, logic_function, num_vars, suggested_threshold, detail_level
        )

        # Print final model explanation
        print_final_model_explanation_modular(
            final_weights, 0, suggested_threshold, num_vars, accuracy, logic_function, detail_level
        )

    else:
        # Full perceptron with learning rate and bias
        print("\nRunning full perceptron with learning rate and bias")
        
        # Call the perceptron learning function
        final_weights, epochs, history, final_bias = perceptron_learning(
            logic_function,
            num_vars,
            learning_rate=1.0,
            threshold=suggested_threshold,
            bias_init=suggested_bias,
            weights_init=None,
            max_epochs=max_epochs,
            skip_zeros=suggested_skip,
            show_details=detail_level == "detailed"
        )

        # Format weights for display
        weights_display = ", ".join([f"{w:.2f}" for w in final_weights])

        print("\n" + "=" * 50)
        print("TRAINING SUMMARY")
        print("=" * 50)
        print(f"Training completed in {epochs} epochs")
        print(f"Final weights: [{weights_display}]")
        print(f"Final bias: {final_bias:.2f}")

        # Verify the model
        accuracy = print_verification_modular(
            final_weights, final_bias, logic_function, num_vars, suggested_threshold, detail_level
        )

        # Print final model explanation
        print_final_model_explanation_modular(
            final_weights, final_bias, suggested_threshold, num_vars, accuracy, 
            logic_function, detail_level
        )


def main():
    """Run demonstrations for various logic functions."""
    print("=" * 70)
    print("PERCEPTRON LEARNING ALGORITHM DEMONSTRATIONS")
    print("=" * 70)
    print("\nThis demo will show how a perceptron learns different logic functions.")
    
    # Define the functions to demonstrate
    demo_functions = [
        ("a and b", False),           # AND function with full perceptron
        ("a or b", False),            # OR function with full perceptron
        ("not a", False),             # NOT function with full perceptron
        ("not (a and b)", False),     # NAND function with full perceptron
        ("a != b", False),            # XOR function (will fail with single perceptron)
        ("(a or b) and c", False),    # 3-variable function with full perceptron
        ("a and b", True)             # AND function with simple perceptron
    ]
    
    # Run each demonstration
    for i, (func, simple_mode) in enumerate(demo_functions):
        run_demo(func, simple_mode)
        if i < len(demo_functions) - 1:
            input("\nPress Enter to continue to the next demonstration...")
    
    print("\n" + "=" * 70)
    print("                END OF DEMONSTRATIONS")
    print("=" * 70)


if __name__ == "__main__":
    main()