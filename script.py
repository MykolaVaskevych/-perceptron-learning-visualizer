"""
Perceptron Learning Algorithm Simulator

A simple implementation of the perceptron learning algorithm for educational purposes.
This simulator demonstrates how a single perceptron learns to approximate binary logic functions.
"""

from utils import parse_logic_function, suggest_initial_parameters
from visualization import (
    print_modular_explanation,
    print_verification_modular,
    print_final_model_explanation_modular
)
from perceptron import simple_perceptron_learning, perceptron_learning


def main():
    """Main function to run the perceptron simulator."""
    # Get logic function from user
    print("=" * 70)
    print("                 PERCEPTRON LEARNING ALGORITHM")
    print("=" * 70)

    # Get explanation detail level from user
    print("\nChoose explanation detail level:")
    print("1. None - Just show the simulation results")
    print("2. Basic - Brief explanations")
    print("3. Detailed - Full step-by-step explanations")
    detail_choice = input("Choose detail level (1-3, default 3): ") or "3"

    if detail_choice == "1":
        detail_level = "none"
    elif detail_choice == "2":
        detail_level = "basic"
    else:
        detail_level = "detailed"

    # Show algorithm explanation based on chosen detail level
    print_modular_explanation(detail_level)

    print("\nEnter a logical function using Python syntax with variables (a, b, c, ...)")
    print("Examples:")
    print("  - not a          (NOT function)")
    print("  - a and b        (AND function)")
    print("  - a or b         (OR function)")
    print("  - a != b         (XOR function - this one is tricky for perceptrons!)")
    print("  - not (a and b)  (NAND function)")
    print("  - not (a or b)   (NOR function)")
    print("  - (a or b) and c (3-input function)")

    logic_function = input("\nEnter logical function: ")

    # Parse the function to determine the number of variables
    try:
        num_vars, variables = parse_logic_function(logic_function)
        print(f"Detected variables: {', '.join(variables)} (total: {num_vars})")
    except ValueError as e:
        print(f"Error: {e}")
        num_vars = int(input("Enter the number of variables manually: "))

    # Ask if user wants to use simple mode (no learning rate or bias)
    use_simple_mode = input("\nUse simple mode without learning rate and bias? (y/n, default n): ").lower() == 'y'

    if use_simple_mode:
        # Simple mode configuration
        print("\nCONFIGURING SIMPLE PERCEPTRON (No learning rate or bias)")

        # Get suggested parameters based on function type
        suggested_threshold, _, _, _ = suggest_initial_parameters(logic_function, num_vars)

        print(f"\nSuggested threshold for '{logic_function}': {suggested_threshold}")
        use_suggested_threshold = input("Use this suggested threshold? (y/n, default y): ").lower() != 'n'

        if use_suggested_threshold:
            threshold = suggested_threshold
        else:
            threshold = float(input(f"Enter threshold (default {suggested_threshold}): ") or suggested_threshold)

        # Ask for initial weights
        weights_init_str = input(
            f"\nEnter initial weights for {num_vars} variables (comma-separated, default all zeros): ")
        weights_init = None
        if weights_init_str:
            try:
                weights_init = [float(w.strip()) for w in weights_init_str.split(",")]
                if len(weights_init) != num_vars:
                    print(
                        f"Warning: {len(weights_init)} weights provided for {num_vars} variables. Using zeros instead.")
                    weights_init = None
            except ValueError:
                print("Invalid weights format. Using zeros instead.")

        max_epochs = int(input("\nEnter maximum epochs (default 100): ") or 100)

        # Run simple perceptron learning
        print(f"\nTraining simple perceptron for logic function: {logic_function}")
        final_weights, epochs, history = simple_perceptron_learning(
            logic_function,
            num_vars,
            threshold,
            weights_init,
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
        accuracy = print_verification_modular(final_weights, 0, logic_function, num_vars, threshold, detail_level)

        # Print final model explanation
        print_final_model_explanation_modular(final_weights, 0, threshold, num_vars, accuracy, logic_function,
                                           detail_level)

    else:
        # Full perceptron configuration with learning rate and bias
        # Get suggested parameters based on function type
        suggested_threshold, suggested_bias, suggested_skip, _ = suggest_initial_parameters(
            logic_function, num_vars
        )

        # Ask user whether to use suggested parameters
        print(f"\nSuggested parameters for '{logic_function}':")
        print(f"- Threshold: {suggested_threshold}")
        print(f"- Initial bias: {suggested_bias}")
        print(f"- Skip zeros input: {'Yes' if suggested_skip else 'No'}")

        use_suggested = input("Use these suggested parameters? (y/n, default y): ").lower() != 'n'

        if use_suggested:
            threshold = suggested_threshold
            bias_init = suggested_bias
            skip_zeros = suggested_skip
            learning_rate = 1.0
            print("Using suggested parameters.")
        else:
            # Get learning parameters from user with explanations
            if detail_level != "none":
                print("\nLearning Rate: Controls how quickly weights are adjusted (typically between 0.1 and 1)")
                print("- Higher values (e.g., 1.0): Faster learning but might overshoot")
                print("- Lower values (e.g., 0.1): Slower, more careful learning")

            learning_rate = float(input("Enter learning rate (default 1): ") or 1)

            if detail_level != "none":
                print("\nChoose Threshold Method:")
                print("1. Manual: You specify the threshold value")
                print("2. Auto-calculate: Optimal threshold is determined automatically")

            threshold_choice = input("Choose threshold method (1 or 2, default 2): ") or "2"

            if threshold_choice == "1":
                if detail_level != "none":
                    print("\nThreshold: The value that activation must meet or exceed for output to be 1")
                    print("- Recommended values:")
                    print(f"  • AND function: around {num_vars / 2 + 0.5:.1f}-{num_vars:.1f}")
                    print("  • OR function: around 0.5")
                    print("  • NOT function: around 0.5")

                threshold = float(input(f"Enter threshold (default {suggested_threshold}): ") or suggested_threshold)
            else:
                threshold = None  # Will be auto-calculated
                print("Threshold will be automatically calculated for optimal separation.")

            if detail_level != "none":
                print("\nInitial Bias: Starting value for the bias term")
                print("- This shifts the decision boundary away from the origin")
                print("- Negative values make it harder to produce 1 as output")
                print(f"- Recommended for '{logic_function}': {suggested_bias}")

            bias_init = float(input(f"Enter initial bias (default {suggested_bias}): ") or suggested_bias)

            if detail_level != "none":
                print("\nSkip All-Zeros Input: Whether to include the input where all variables are 0")
                print("- Including [0,0,...] ensures the perceptron learns the complete function")
                print("- For OR functions, it's important to include [0,0,...] to get correct behavior")

            skip_zeros_input = input("Skip all-zeros input during training? (y/n, default n): ").lower() == 'y'
            skip_zeros = skip_zeros_input

        # Ask for initial weights
        if detail_level != "none":
            print("\nInitial Weights: Starting values for each input weight")
            print(f"- Enter {num_vars} comma-separated values (one for each variable)")
            print("- Default is all zeros, which means no inputs have influence initially")

        weights_init_str = input(
            f"Enter initial weights for {num_vars} variables (comma-separated, default all zeros): ")
        weights_init = None
        if weights_init_str:
            try:
                weights_init = [float(w.strip()) for w in weights_init_str.split(",")]
                if len(weights_init) != num_vars:
                    print(
                        f"Warning: {len(weights_init)} weights provided for {num_vars} variables. Using zeros instead.")
                    weights_init = None
            except ValueError:
                print("Invalid weights format. Using zeros instead.")

        max_epochs = int(input("\nEnter maximum epochs (default 100): ") or 100)

        # Call the perceptron learning function
        print(f"\nTraining perceptron for logic function: {logic_function}")
        final_weights, epochs, history, final_bias = perceptron_learning(
            logic_function,
            num_vars,
            learning_rate,
            threshold,
            bias_init,
            weights_init,
            max_epochs,
            skip_zeros,
            detail_level == "detailed"  # Only show detailed calculations if detailed mode
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
        if threshold is None:
            # If threshold was auto-calculated, recalculate it for final weights
            from utils import calculate_optimal_threshold
            import itertools
            import numpy as np
            # Generate all possible binary inputs
            input_combinations = list(itertools.product([0, 1], repeat=num_vars))
            inputs = np.array(input_combinations)
            # Calculate target outputs for each input
            var_names = [chr(ord('a') + i) for i in range(num_vars)]
            from utils import calculate_target
            targets = np.array([calculate_target(inp, logic_function, var_names) for inp in inputs])
            threshold = calculate_optimal_threshold(inputs, targets, final_weights, final_bias)
            print(f"\nFinal calculated threshold: {threshold:.2f}")

        accuracy = print_verification_modular(final_weights, final_bias, logic_function, num_vars, threshold,
                                           detail_level)

        # Print final model explanation
        print_final_model_explanation_modular(final_weights, final_bias, threshold, num_vars, accuracy, logic_function,
                                           detail_level)

    print("\n" + "=" * 70)
    print("                   END OF SIMULATION")
    print("=" * 70)


if __name__ == "__main__":
    main()