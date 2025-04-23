import ast

def parse_env_kwargs(sys_argv):
    """
    Parses environment-specific arguments from the command-line arguments.

    This function extracts arguments that start with the "--env." prefix, converts their values to
    appropriate types (int, float, or string), and removes them from the argument list to avoid
    duplication when passing to other parsers.

    Args:
        sys_argv (list of str): The list of command-line arguments, typically sys.argv.

    Returns:
        tuple:
            - dict: A dictionary of parsed environment arguments with converted values.
            - list: The modified list of command-line arguments with parsed arguments removed.

    Notes:
        - If a value can be converted to an int, it is stored as an int.
        - If a value can be converted to a float, it is stored as a float.
        - Otherwise, the value remains a string.
        - If an argument is missing a value, a warning is printed, and the argument is ignored.
    """
    env_arg_prefix = "--env."
    env_kwargs = {}
    modified_sys_argv = []
    i = 0
    while i < len(sys_argv):
        arg = sys_argv[i]
        if arg.startswith(env_arg_prefix):
            arg_name = arg[len(env_arg_prefix):]
            if i + 1 < len(sys_argv):
                value_str = sys_argv[i + 1]  # Get the value as string

                try:
                    # Attempt to parse list, dict, tuple, string, number, boolean safely
                    value = ast.literal_eval(value_str)
                    env_kwargs[arg_name] = value
                except (ValueError, SyntaxError): # If parsing fails, treat as string
                    env_kwargs[arg_name] = value_str
                i += 2 # Increment i by 2 to skip both the argument and its value
            else:
                print(f"Warning: Argument '{arg}' is missing a value and will be ignored.")
                i += 1 # Increment i to skip the argument without value
        else:
            modified_sys_argv.append(arg)
            i += 1 # Increment i to the next argument

    return env_kwargs, modified_sys_argv
