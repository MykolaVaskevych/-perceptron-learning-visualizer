"""
Visualization functions for displaying perceptron results.
"""

from tabulate import tabulate
from explanations import BASIC_EXPLANATION, DETAILED_EXPLANATION


def print_modular_explanation(detail_level="basic"):
    """
    Prints a modular, beginner-friendly explanation of the perceptron learning algorithm.
    The level of detail can be controlled.

    Args:
        detail_level: String indicating the level of detail ("none", "basic", "detailed")
    """
    if detail_level == "none":
        return

    # Basic header always shown if not "none"
    print("\n" + "=" * 70)
    print("                 PERCEPTRON LEARNING ALGORITHM")
    print("=" * 70)

    if detail_level == "basic":
        print(BASIC_EXPLANATION)
        return

    # Full detailed explanation
    print(DETAILED_EXPLANATION)


def print_verification_modular(final_weights, final_bias, logic_function, num_vars, threshold, detail_level="detailed"):
    """Prints verification of the model using the final weights and bias, with adjustable detail level."""
    if detail_level == "none":
        return

    print("\n" + "=" * 50)
    print("VERIFICATION OF TRAINED PERCEPTRON")
    print("=" * 50)

    # Import here to avoid circular imports
    import numpy as np
    import itertools
    from utils import calculate_target

    if detail_level == "basic":
        print("Testing if our trained perceptron correctly predicts all possible inputs.")

        # Generate variable names and input combinations
        var_names = [chr(ord('a') + i) for i in range(num_vars)]

        input_combinations = list(itertools.product([0, 1], repeat=num_vars))
        inputs = np.array(input_combinations)

        # Simplified verification table
        verification_table = []
        verification_header = ["Input", "Output", "Expected", "Correct?"]

        correct_count = 0
        for inp in inputs:
            # Calculate output
            activation = np.dot(inp, final_weights) + final_bias
            output = 1 if activation >= threshold else 0

            # Calculate expected output
            expected = calculate_target(inp, logic_function, var_names)

            # Check if correct
            is_correct = output == expected
            if is_correct:
                correct_count += 1
                correct_str = "✓"
            else:
                correct_str = "✗"

            # Add row to verification table
            verification_row = [str(inp.tolist()), output, expected, correct_str]
            verification_table.append(verification_row)

        # Display verification table
        print(tabulate(verification_table, headers=verification_header, tablefmt="grid"))

        # Display accuracy
        accuracy = correct_count / len(inputs) * 100
        print(f"\nAccuracy: {correct_count}/{len(inputs)} correct predictions ({accuracy:.1f}%)")

        return accuracy

    # Detailed verification (original implementation)
    print("Let's verify if our trained perceptron correctly predicts all possible inputs.")
    print("We'll compare our perceptron's output with the expected output from the logic function.")

    # Generate variable names and input combinations
    var_names = [chr(ord('a') + i) for i in range(num_vars)]

    input_combinations = list(itertools.product([0, 1], repeat=num_vars))
    inputs = np.array(input_combinations)

    # Prepare verification table
    verification_table = []
    verification_header = ["Input", "Detailed Calculation", "Perceptron Output", "Expected Output", "Correct?"]

    correct_count = 0

    for inp in inputs:
        # Calculate activation with final weights and bias
        activation = np.dot(inp, final_weights) + final_bias
        output = 1 if activation >= threshold else 0

        # Create detailed calculation
        detailed_calc = []
        for j, val in enumerate(inp):
            if val != 0:  # Only show non-zero terms
                detailed_calc.append(f"{val} × {final_weights[j]:.2f} = {val * final_weights[j]:.2f}")

        if detailed_calc:
            detailed_calc_str = " + ".join(detailed_calc)
            if final_bias != 0:
                detailed_calc_str += f" + {final_bias:.2f}"
            detailed_calc_str += f" = {activation:.2f}"
        else:
            detailed_calc_str = f"{final_bias:.2f} = {activation:.2f}"

        # Add threshold comparison with clear explanation
        if activation >= threshold:
            detailed_calc_str += f" ≥ {threshold:.2f}, so output = 1"
        else:
            detailed_calc_str += f" < {threshold:.2f}, so output = 0"

        # Calculate expected output
        expected = calculate_target(inp, logic_function, var_names)

        # Check if correct
        is_correct = output == expected
        if is_correct:
            correct_count += 1
            correct_str = "✓"
        else:
            correct_str = "✗"

        # Add row to verification table
        verification_row = [str(inp.tolist()), detailed_calc_str, output, expected, correct_str]
        verification_table.append(verification_row)

    # Display verification table
    print(tabulate(verification_table, headers=verification_header, tablefmt="grid"))

    # Display accuracy
    accuracy = correct_count / len(inputs) * 100
    print(f"\nAccuracy: {correct_count}/{len(inputs)} correct predictions ({accuracy:.1f}%)")

    return accuracy


def print_final_model_explanation_modular(final_weights, final_bias, threshold, num_vars, accuracy, logic_function,
                                      detail_level="detailed"):
    """Prints a clear explanation of the final perceptron model, with adjustable detail level."""
    if detail_level == "none":
        return

    print("\n" + "=" * 50)
    print("FINAL PERCEPTRON MODEL")
    print("=" * 50)

    # Generate variable names
    var_names = [chr(ord('a') + i) for i in range(num_vars)]

    # Create a mathematical formula representation
    formula_parts = []
    for j in range(num_vars):
        var_name = var_names[j]
        formula_parts.append(f"{final_weights[j]:.2f}×{var_name}")

    formula = " + ".join(formula_parts)
    if final_bias != 0:
        if final_bias > 0:
            formula += f" + {final_bias:.2f}"
        else:
            formula += f" - {abs(final_bias):.2f}"

    # Basic explanation
    if detail_level == "basic":
        print(f"Formula: Activation = {formula}")
        print(f"Output = 1 if Activation ≥ {threshold:.2f}, otherwise Output = 0")
        print(f"Accuracy: {accuracy:.1f}%")

        if accuracy < 100:
            print("\n⚠ Note: The perceptron couldn't achieve 100% accuracy.")
            if "xor" in logic_function.lower() or "!=" in logic_function.lower() or "^" in logic_function.lower():
                print("This is because XOR is not linearly separable and requires a multi-layer network.")
        return

    # Detailed explanation
    print("The perceptron has learned the following model:")
    print("\n1. MATHEMATICAL FORMULA:")
    print(f"   Activation = {formula}")
    print(f"   Output = 1 if Activation ≥ {threshold:.2f}, otherwise Output = 0")

    print("\n2. DECISION PROCESS:")
    print(f"   a) Multiply each input by its corresponding weight")
    for j in range(num_vars):
        print(f"      • When {var_names[j]}=1: Add {final_weights[j]:.2f} to the activation")
    print(f"   b) Add the bias: {final_bias:.2f}")
    print(f"   c) Compare the result to the threshold ({threshold:.2f})")
    print(f"   d) Output 1 if greater than or equal to threshold, otherwise output 0")

    print("\n3. WEIGHTS INTERPRETATION:")
    print("   Positive weights strengthen the activation when their input is 1.")
    print("   Negative weights weaken the activation when their input is 1.")
    print("   Larger absolute values indicate greater importance of that input.")

    for j in range(num_vars):
        weight = final_weights[j]
        var_name = var_names[j]
        if weight > 0:
            print(f"   • {var_name}: Weight = {weight:.2f} (Positive) → When {var_name}=1, increases activation")
        elif weight < 0:
            print(f"   • {var_name}: Weight = {weight:.2f} (Negative) → When {var_name}=1, decreases activation")
        else:
            print(f"   • {var_name}: Weight = {weight:.2f} (Zero) → This input has no effect")

    print(f"\n   • Bias: {final_bias:.2f} → " +
          ("Increases activation regardless of inputs" if final_bias > 0 else
           "Decreases activation regardless of inputs" if final_bias < 0 else
           "Has no effect on activation"))

    print("\n4. PERFORMANCE:")
    print(f"   Accuracy: {accuracy:.1f}%")

    if accuracy < 100:
        print("\n   ⚠ Note: The perceptron couldn't achieve 100% accuracy.")

        # Special explanation for OR function with high bias
        if "a or b" in logic_function.lower() and final_bias > threshold and accuracy == 75:
            print("   This is because the bias is too high, causing the perceptron to output 1")
            print("   even when all inputs are 0 (which should output 0 for an OR function).")
            print("   To fix this, include the [0,0] case during training or use a lower initial bias.")
        elif "a != b" in logic_function.lower() or "a ^ b" in logic_function.lower() or "xor" in logic_function.lower():
            print("   This is because XOR is not linearly separable, which means a single perceptron")
            print("   cannot learn this function perfectly. You would need a multi-layer network.")
        else:
            print("   This may indicate that the function is not linearly separable,")
            print("   which a single perceptron cannot learn perfectly.")
    else:
        print("\n   ✓ Perfect accuracy! The perceptron has successfully learned the function.")