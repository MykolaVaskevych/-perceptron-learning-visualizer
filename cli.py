"""
Command-line interface for the perceptron learning algorithm.
"""

import argparse
from perceptron import perceptron_learning, simple_perceptron_learning
from visualization import (
    print_modular_explanation,
    print_verification_modular,
    print_final_model_explanation_modular
)
from utils import parse_logic_function, suggest_initial_parameters


def main():
    """Handle command-line arguments and run the perceptron learning algorithm."""
    parser = argparse.ArgumentParser(
        description="Run the perceptron learning algorithm on a logical function."
    )
    
    parser.add_argument(
        "--function", "-f", type=str, required=True,
        help="The logical function to learn, e.g., 'a and b', 'a or b', 'not a', 'a != b'"
    )
    
    parser.add_argument(
        "--simple", "-s", action="store_true",
        help="Use simple perceptron mode (no bias or learning rate)"
    )
    
    parser.add_argument(
        "--detail", "-d", type=str, choices=["none", "basic", "detailed"], default="basic",
        help="Level of detail for explanations"
    )
    
    parser.add_argument(
        "--epochs", "-e", type=int, default=100,
        help="Maximum number of training epochs"
    )
    
    parser.add_argument(
        "--learning-rate", "-lr", type=float, default=1.0,
        help="Learning rate for weight updates (only used in full perceptron mode)"
    )
    
    parser.add_argument(
        "--threshold", "-t", type=float,
        help="Activation threshold (if not provided, a suggested value will be used)"
    )
    
    parser.add_argument(
        "--bias", "-b", type=float,
        help="Initial bias value (if not provided, a suggested value will be used)"
    )
    
    parser.add_argument(
        "--weights", "-w", type=str,
        help="Comma-separated initial weights (if not provided, zeros will be used)"
    )
    
    parser.add_argument(
        "--skip-zeros", action="store_true",
        help="Skip all-zeros input during training"
    )
    
    parser.add_argument(
        "--explain", action="store_true",
        help="Show an explanation of the perceptron learning algorithm"
    )
    
    args = parser.parse_args()
    
    # Process the logical function
    try:
        num_vars, variables = parse_logic_function(args.function)
        print(f"Detected variables: {', '.join(variables)} (total: {num_vars})")
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Show explanation if requested
    if args.explain:
        print_modular_explanation(args.detail)
    
    # Get suggested parameters based on function type
    suggested_threshold, suggested_bias, suggested_skip, _ = suggest_initial_parameters(
        args.function, num_vars
    )
    
    # Process weight initialization
    weights_init = None
    if args.weights:
        try:
            weights_init = [float(w.strip()) for w in args.weights.split(",")]
            if len(weights_init) != num_vars:
                print(
                    f"Warning: {len(weights_init)} weights provided for {num_vars} variables. Using zeros instead."
                )
                weights_init = None
        except ValueError:
            print("Invalid weights format. Using zeros instead.")
    
    if args.simple:
        # Simple perceptron mode (no bias or learning rate)
        threshold = args.threshold if args.threshold is not None else suggested_threshold
        
        print(f"\nRunning simple perceptron for function: {args.function}")
        print(f"Using threshold: {threshold}")
        
        final_weights, epochs, history = simple_perceptron_learning(
            args.function,
            num_vars,
            threshold,
            weights_init,
            args.epochs,
            args.detail == "detailed"
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
            final_weights, 0, args.function, num_vars, threshold, args.detail
        )
        
        # Print final model explanation
        print_final_model_explanation_modular(
            final_weights, 0, threshold, num_vars, accuracy, args.function, args.detail
        )
    
    else:
        # Full perceptron mode with bias and learning rate
        threshold = args.threshold if args.threshold is not None else suggested_threshold
        bias_init = args.bias if args.bias is not None else suggested_bias
        skip_zeros = args.skip_zeros if args.skip_zeros else suggested_skip
        
        print(f"\nRunning full perceptron for function: {args.function}")
        print(f"Using threshold: {threshold}, bias: {bias_init}, learning rate: {args.learning_rate}")
        
        final_weights, epochs, history, final_bias = perceptron_learning(
            args.function,
            num_vars,
            args.learning_rate,
            threshold,
            bias_init,
            weights_init,
            args.epochs,
            skip_zeros,
            args.detail == "detailed"
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
            final_weights, final_bias, args.function, num_vars, threshold, args.detail
        )
        
        # Print final model explanation
        print_final_model_explanation_modular(
            final_weights, final_bias, threshold, num_vars, accuracy, args.function, args.detail
        )


if __name__ == "__main__":
    main()