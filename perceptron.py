"""
Perceptron learning algorithm implementations.
"""

import numpy as np
import itertools
from tabulate import tabulate
from utils import calculate_target, suggest_initial_parameters


def simple_perceptron_learning(logic_function, num_vars=2, threshold=0.5, weights_init=None,
                               max_epochs=100, show_details=True):
    """
    A simplified perceptron learning algorithm without learning rate or bias.
    Useful for educational purposes and understanding the core concept.

    Args:
        logic_function: A string representing the logic function (e.g., "a and b")
        num_vars: Number of input variables (default 2)
        threshold: The activation threshold (default 0.5)
        weights_init: Initial weights (default None, will be initialized to zeros)
        max_epochs: Maximum number of epochs to train (default 100)
        show_details: Whether to show detailed calculations (default True)

    Returns:
        weights: Final weight vector
        epochs: Number of epochs until convergence
        history: History of weights and decisions during training
    """
    # Generate variable names (a, b, c, ...)
    var_names = [chr(ord('a') + i) for i in range(num_vars)]

    # Initialize weights
    if weights_init is None:
        weights = np.zeros(num_vars)
        weights_str = "All zeros"
    else:
        weights = np.array(weights_init)
        if len(weights) != num_vars:
            raise ValueError(f"weights_init must have length {num_vars}")
        weights_str = ", ".join([f"{w:.2f}" for w in weights])

    # Generate all possible binary inputs
    input_combinations = list(itertools.product([0, 1], repeat=num_vars))
    inputs = np.array(input_combinations)

    # Calculate target outputs for each input
    targets = np.array([calculate_target(inp, logic_function, var_names) for inp in inputs])

    # Display the initial configuration
    if show_details:
        print("\n" + "=" * 50)
        print("SIMPLE PERCEPTRON CONFIGURATION")
        print("=" * 50)

        config_table = [
            ["Logic Function", logic_function],
            ["Number of Variables", num_vars],
            ["Variable Names", ", ".join(var_names)],
            ["Threshold", f"{threshold:.2f}"],
            ["Initial Weights", weights_str],
            ["Max Epochs", max_epochs]
        ]

        print(tabulate(config_table, tablefmt="grid"))
        print("\n")

        # Display truth table for the function
        print("\n" + "=" * 50)
        print(f"TRUTH TABLE FOR: {logic_function}")
        print("=" * 50)
        print("This table shows all possible inputs and the expected outputs for the logical function.")

        truth_table = []
        header = var_names + ["Expected Output"]

        for i, inp in enumerate(inputs):
            row = inp.tolist() + [targets[i]]
            truth_table.append(row)

        print(tabulate(truth_table, headers=header, tablefmt="grid"))

        print("\nThe perceptron will learn to approximate this truth table by adjusting its weights.")

    # Training process
    if show_details:
        print("\n" + "=" * 50)
        print("SIMPLIFIED PERCEPTRON TRAINING PROCESS")
        print("=" * 50)
        print(
            "We'll train the perceptron by presenting input patterns and adjusting weights when output doesn't match target.")
        print("This simplified version doesn't use learning rate or bias.")

    history = []
    epoch = 1
    weights_changed = True

    while weights_changed and epoch <= max_epochs:
        weights_changed = False

        if show_details:
            print(f"\n--- EPOCH {epoch} ---")
            print(f"Starting with weights: [{', '.join([f'{w:.2f}' for w in weights])}]")

            # Prepare headers for the training table
            training_header = ["Step", "Input", "Weights", "Detailed Calculation", "Output", "Target", "Update"]
            training_table = []

        step = 1
        # Process inputs one by one
        for i, inp in enumerate(inputs):
            # Calculate activation (no bias)
            activation = np.dot(inp, weights)

            # Apply threshold function
            output = 1 if activation >= threshold else 0

            # Format current weights for display
            weights_display = ", ".join([f"{w:.2f}" for w in weights])

            # Create calculation string for detailed display
            if show_details:
                detailed_calc = []
                for j, val in enumerate(inp):
                    if val != 0:  # Only show non-zero terms
                        detailed_calc.append(f"{val} × {weights[j]:.2f} = {val * weights[j]:.2f}")

                if detailed_calc:
                    detailed_calc_str = " + ".join(detailed_calc) + f" = {activation:.2f}"
                else:
                    detailed_calc_str = f"0 = {activation:.2f}"

                # Add threshold comparison with clear explanation
                if activation >= threshold:
                    detailed_calc_str += f" ≥ {threshold:.2f}, so output = 1"
                else:
                    detailed_calc_str += f" < {threshold:.2f}, so output = 0"

            # Record current state before any updates
            current_state = {
                'epoch': epoch,
                'step': step,
                'input': inp.tolist(),
                'weights': weights.copy(),
                'activation': activation,
                'output': output,
                'target': targets[i],
                'weight_update': 'none'
            }

            # Default update display
            update_display = "No update needed (output matches target)"

            # Check if output matches target
            if output != targets[i]:
                # Update weights based on error (simplified approach: add or subtract input values)
                # Direction of update depends on target
                delta = 1 if targets[i] > output else -1
                weight_update = delta * inp

                # Calculate new weights
                new_weights = weights + weight_update
                weights_changed = True

                # Prepare update string for display
                detailed_updates = []
                for j, val in enumerate(inp):
                    if val != 0:  # Only update for non-zero inputs
                        detailed_updates.append(
                            f"w{j + 1} = {weights[j]:.2f} + ({delta} × {val}) = {new_weights[j]:.2f}")

                # Store new values in state
                current_state['new_weights'] = new_weights.copy()
                current_state['weight_update'] = 'updated'
                current_state['detailed_updates'] = detailed_updates

                # Format update display for detailed mode
                if show_details:
                    update_display = f"Error detected! Output ({output}) ≠ Target ({targets[i]})\n"
                    update_display += f"delta = {delta} (direction based on target)\n"
                    # Show detailed weight updates
                    update_display += "\n".join(detailed_updates) + "\n"
                    update_display += f"New weights: [{', '.join([f'{w:.2f}' for w in new_weights])}]"

                # Actually update the weights for next step
                weights = new_weights

            # Create training table row if showing details
            if show_details:
                inp_str = str(inp.tolist())
                training_row = [
                    step,
                    inp_str,
                    f"[{weights_display}]",
                    detailed_calc_str,
                    output,
                    targets[i],
                    update_display
                ]
                training_table.append(training_row)

            # Save state to history
            history.append(current_state)
            step += 1

        # Display the training table for this epoch if showing details
        if show_details:
            print(tabulate(training_table, headers=training_header, tablefmt="grid"))

            # If weights didn't change in this epoch, we're done
            if not weights_changed:
                print("\n✓ Converged! No weight changes in this epoch.")
            elif epoch == max_epochs:
                print(f"\n⚠ Reached maximum number of epochs ({max_epochs}) without convergence.")

        epoch += 1

    return weights, epoch - 1, history


def perceptron_learning(logic_function, num_vars=2, learning_rate=1, threshold=None, bias_init=None, weights_init=None,
                        max_epochs=100, skip_zeros=False, show_details=True):
    """
    Implements the perceptron learning algorithm for a binary logic function with n variables.

    Args:
        logic_function: A string representing the logic function (e.g., "(not a) and b")
        num_vars: Number of input variables (default 2)
        learning_rate: The learning rate for weight updates (default 1)
        threshold: The activation threshold (default None, will be auto-calculated)
        bias_init: Initial bias value (default None, will use suggested value)
        weights_init: Initial weights (default None, will be initialized to zeros)
        max_epochs: Maximum number of epochs to train (default 100)
        skip_zeros: Whether to skip the all-zeros input during training (default False)
        show_details: Whether to show detailed calculations (default True)

    Returns:
        weights: Final weight vector
        epochs: Number of epochs until convergence
        history: History of weights and decisions during training
        bias: Final bias term for the perceptron
    """
    # Generate variable names (a, b, c, ...)
    var_names = [chr(ord('a') + i) for i in range(num_vars)]

    # Initialize weights
    if weights_init is None:
        weights = np.zeros(num_vars)
        weights_str = "All zeros"
    else:
        weights = np.array(weights_init)
        if len(weights) != num_vars:
            raise ValueError(f"weights_init must have length {num_vars}")
        weights_str = ", ".join([f"{w:.2f}" for w in weights])

    # Generate all possible binary inputs
    input_combinations = list(itertools.product([0, 1], repeat=num_vars))
    inputs = np.array(input_combinations)

    # Calculate target outputs for each input
    targets = np.array([calculate_target(inp, logic_function, var_names) for inp in inputs])

    # If bias_init is None, use a suggested value
    if bias_init is None:
        suggested_threshold, suggested_bias, _, _ = suggest_initial_parameters(logic_function, num_vars, weights_init)
        bias_init = suggested_bias

    # Initialize bias term
    bias = bias_init

    # If threshold is None, auto-calculate it
    if threshold is None:
        from utils import calculate_optimal_threshold
        threshold = calculate_optimal_threshold(inputs, targets, weights, bias)
        if show_details:
            print(f"\nAutomatically calculated threshold: {threshold:.2f}")

    # Display the initial configuration
    if show_details:
        print("\n" + "=" * 50)
        print("PERCEPTRON CONFIGURATION")
        print("=" * 50)

        config_table = [
            ["Logic Function", logic_function],
            ["Number of Variables", num_vars],
            ["Variable Names", ", ".join(var_names)],
            ["Learning Rate", learning_rate],
            ["Threshold", f"{threshold:.2f}" + (" (auto-calculated)" if threshold is None else "")],
            ["Initial Bias", bias_init],
            ["Initial Weights", weights_str],
            ["Skip All-Zeros Input", "Yes" if skip_zeros else "No"],
            ["Max Epochs", max_epochs]
        ]

        print(tabulate(config_table, tablefmt="grid"))
        print("\n")

        # Display truth table for the function
        print("\n" + "=" * 50)
        print(f"TRUTH TABLE FOR: {logic_function}")
        print("=" * 50)
        print("This table shows all possible inputs and the expected outputs for the logical function.")

        truth_table = []
        header = var_names + ["Expected Output"]

        for i, inp in enumerate(inputs):
            row = inp.tolist() + [targets[i]]
            truth_table.append(row)

        print(tabulate(truth_table, headers=header, tablefmt="grid"))

        print("\nThe perceptron will learn to approximate this truth table by adjusting its weights and bias.")

    # Training process
    if show_details:
        print("\n" + "=" * 50)
        print("PERCEPTRON TRAINING PROCESS")
        print("=" * 50)
        print(
            "We'll train the perceptron by presenting input patterns and adjusting weights when output doesn't match target.")
        print("The training continues until either weights stop changing or we reach the maximum number of epochs.")

    history = []
    epoch = 1
    weights_changed = True

    while weights_changed and epoch <= max_epochs:
        weights_changed = False

        if show_details:
            print(f"\n--- EPOCH {epoch} ---")
            print(f"Starting with weights: [{', '.join([f'{w:.2f}' for w in weights])}], bias: {bias:.2f}")

            # Prepare headers for the training table
            training_header = ["Step", "Input", "Weights", "Bias", "Detailed Calculation", "Output", "Target", "Update"]
            training_table = []

        step = 1
        # Process inputs one by one
        for i, inp in enumerate(inputs):
            # Skip all-zeros input if requested
            if skip_zeros and np.all(inp == 0):
                continue

            # Calculate activation (include bias term)
            activation = np.dot(inp, weights) + bias

            # Apply threshold function
            output = 1 if activation >= threshold else 0

            # Format current weights for display
            weights_display = ", ".join([f"{w:.2f}" for w in weights])

            # Create calculation string for display
            if show_details:
                detailed_calc = []
                for j, val in enumerate(inp):
                    if val != 0:  # Only show non-zero terms
                        detailed_calc.append(f"{val} × {weights[j]:.2f} = {val * weights[j]:.2f}")

                if detailed_calc:
                    detailed_calc_str = " + ".join(detailed_calc)
                    if bias != 0:
                        detailed_calc_str += f" + {bias:.2f}"
                    detailed_calc_str += f" = {activation:.2f}"
                else:
                    detailed_calc_str = f"{bias:.2f} = {activation:.2f}"

                # Add threshold comparison with clear explanation
                if activation >= threshold:
                    detailed_calc_str += f" ≥ {threshold:.2f}, so output = 1"
                else:
                    detailed_calc_str += f" < {threshold:.2f}, so output = 0"

            # Record current state before any updates
            current_state = {
                'epoch': epoch,
                'step': step,
                'input': inp.tolist(),
                'weights': weights.copy(),
                'bias': bias,
                'activation': activation,
                'output': output,
                'target': targets[i],
                'weight_update': 'none'
            }

            # Default update display
            update_display = "No update needed (output matches target)"

            # Check if output matches target
            if output != targets[i]:
                # Update weights based on error
                delta = learning_rate * (targets[i] - output)
                weight_update = delta * inp

                # Calculate new weights and bias
                new_weights = weights + weight_update
                new_bias = bias + delta
                weights_changed = True

                # Prepare update string for display
                if show_details:
                    detailed_updates = []
                    for j, val in enumerate(inp):
                        detailed_updates.append(
                            f"w{j + 1} = {weights[j]:.2f} + ({delta:.2f} × {val}) = {new_weights[j]:.2f}")

                    # Add detailed bias update calculation
                    detailed_bias_update = f"bias = {bias:.2f} + {delta:.2f} = {new_bias:.2f}"

                    # Format update display
                    update_display = f"Error detected! Output ({output}) ≠ Target ({targets[i]})\n"
                    update_display += f"delta = {delta:.2f} (learning_rate × (target - output))\n"
                    # Show detailed weight updates
                    update_display += "\n".join(detailed_updates) + "\n"
                    update_display += detailed_bias_update + "\n"
                    update_display += f"New weights: [{', '.join([f'{w:.2f}' for w in new_weights])}], New bias: {new_bias:.2f}"

                # Store new values in state
                current_state['new_weights'] = new_weights.copy()
                current_state['new_bias'] = new_bias
                current_state['delta'] = delta
                current_state['weight_update'] = 'updated'

                # Actually update the weights and bias for next step
                weights = new_weights
                bias = new_bias

            # Create training table row
            if show_details:
                inp_str = str(inp.tolist())
                training_row = [
                    step,
                    inp_str,
                    f"[{weights_display}]",
                    f"{bias:.2f}",
                    detailed_calc_str,
                    output,
                    targets[i],
                    update_display
                ]
                training_table.append(training_row)

            # Save state to history
            history.append(current_state)
            step += 1

        # Display the training table for this epoch
        if show_details:
            print(tabulate(training_table, headers=training_header, tablefmt="grid"))

            # If weights didn't change in this epoch, we're done
            if not weights_changed:
                print("\n✓ Converged! No weight changes in this epoch.")
            elif epoch == max_epochs:
                print(f"\n⚠ Reached maximum number of epochs ({max_epochs}) without convergence.")

        epoch += 1

    return weights, epoch - 1, history, bias