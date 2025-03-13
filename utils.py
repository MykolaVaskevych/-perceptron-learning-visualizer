"""
Utility functions for the perceptron implementation.
"""
import re
import numpy as np


def calculate_target(inputs, function_str, var_names):
    """Calculate target output based on the given logic function."""
    # Convert to boolean
    bool_inputs = [bool(inp) for inp in inputs]

    # Create namespace with variables
    namespace = {var_names[i]: bool_inputs[i] for i in range(len(var_names))}
    namespace['result'] = None

    # Prepare the code to execute
    code = f"result = {function_str}"

    try:
        # Execute the code in the namespace
        exec(code, namespace)

        # Get the result
        result = namespace['result']

        # Convert to binary (0 or 1)
        return 1 if result else 0
    except Exception as e:
        var_suggestions = ", ".join(var_names)
        print(f"Error evaluating function '{function_str}': {e}")
        print(f"Please use Python syntax with variables: {var_suggestions}")
        raise ValueError(f"Invalid logical function: {function_str}")


def parse_logic_function(function_str):
    """Analyzes a logic function string to determine the number of variables."""
    # Find all single letter variables (a-z)
    variables = set(re.findall(r'\b([a-z])\b', function_str))

    if not variables:
        raise ValueError("No variables found in the logic function")

    # Map variables to their ASCII code
    var_codes = [ord(var) for var in variables]

    # Determine the number of variables (assuming sequential variables starting from 'a')
    num_vars = max(var_codes) - ord('a') + 1

    return num_vars, sorted(variables)


def calculate_optimal_threshold(inputs, targets, weights, bias):
    """
    Calculates an optimal threshold based on the inputs, targets, weights, and bias.
    Uses the midpoint between highest negative and lowest positive activations.

    Args:
        inputs: Array of input patterns
        targets: Array of target outputs (0 or 1)
        weights: Current weights
        bias: Current bias

    Returns:
        threshold: Calculated optimal threshold
    """
    # Calculate activations for all inputs
    activations = []

    for i, inp in enumerate(inputs):
        activation = np.dot(inp, weights) + bias
        activations.append((activation, targets[i]))

    # Sort activations by value
    activations.sort()

    # Find highest activation for target=0 and lowest for target=1
    highest_negative = None  # Highest activation with target=0
    lowest_positive = None  # Lowest activation with target=1

    for activation, target in activations:
        if target == 0 and (highest_negative is None or activation > highest_negative):
            highest_negative = activation
        if target == 1 and (lowest_positive is None or activation < lowest_positive):
            lowest_positive = activation

    # If we don't have examples of both classes, use default
    if highest_negative is None or lowest_positive is None:
        return 0.5

    # If they overlap (not linearly separable), pick middle point anyway
    if highest_negative >= lowest_positive:
        print("\n⚠ Warning: Function may not be linearly separable with current weights")
        print(f"   Highest activation for target=0: {highest_negative:.2f}")
        print(f"   Lowest activation for target=1: {lowest_positive:.2f}")

    # Calculate midpoint between highest negative and lowest positive
    threshold = (highest_negative + lowest_positive) / 2

    return threshold


def suggest_initial_parameters(logic_function, num_vars, weights_init=None):
    """
    Suggests good initial parameters for the perceptron based on the logical function.
    Uses a more robust parsing approach for complex formulas.

    Args:
        logic_function: String representing logical function
        num_vars: Number of variables
        weights_init: Initial weights if provided

    Returns:
        threshold: Suggested threshold
        bias: Suggested bias
        skip_zeros: Whether to skip all-zeros input
        simple_mode: Whether to use simple mode (no learning rate/bias adjustments)
    """
    logic_lower = logic_function.lower()

    # Count occurrences of different operators
    and_count = logic_lower.count(' and ') + logic_lower.count('&')
    or_count = logic_lower.count(' or ') + logic_lower.count('|')
    not_count = logic_lower.count('not ') + logic_lower.count('!')
    xor_count = logic_lower.count(' != ') + logic_lower.count('^') + logic_lower.count('xor')

    # Check for parentheses to identify complex expressions
    has_complex_nesting = '(' in logic_lower and ')' in logic_lower

    # Default values for simple mode
    simple_mode = False

    # Parse the structure more carefully
    if and_count > 0 and or_count == 0 and xor_count == 0 and not has_complex_nesting:
        # Pure AND function
        threshold = num_vars - 0.5  # Just under the number of inputs
        bias = -0.1
        skip_zeros = False

    elif or_count > 0 and and_count == 0 and xor_count == 0 and not has_complex_nesting:
        # Pure OR function
        threshold = 0.5
        bias = -0.5
        skip_zeros = False

    elif xor_count > 0 and and_count == 0 and or_count == 0:
        # Simple XOR function (note: not linearly separable)
        threshold = 0.5
        bias = 0
        skip_zeros = False
        print(
            "\n⚠ Warning: XOR functions are not linearly separable and cannot be perfectly learned by a single perceptron.")

    elif not_count > 0 and and_count == 0 and or_count == 0 and xor_count == 0:
        # Single NOT function
        threshold = 0.5
        bias = 1.0
        skip_zeros = False

    elif not_count > 0 and and_count > 0 and "not" in logic_lower and "and" in logic_lower and "(" in logic_lower:
        # Possibly a NAND function
        threshold = 0.5
        bias = 1.0
        skip_zeros = False

    elif not_count > 0 and or_count > 0 and "not" in logic_lower and "or" in logic_lower and "(" in logic_lower:
        # Possibly a NOR function
        threshold = 0.5
        bias = 0.5
        skip_zeros = False

    else:
        # Complex or mixed function
        # For complex functions, we'll use more moderate values
        threshold = 0.5
        bias = 0
        skip_zeros = False

        # Warn about potentially non-linearly separable functions
        if (and_count > 0 and or_count > 0) or xor_count > 0 or has_complex_nesting:
            print("\n⚠ Note: Complex logical functions might not be linearly separable.")
            print("   If training fails to converge, consider simplifying the function or using multiple perceptrons.")

    return threshold, bias, skip_zeros, simple_mode