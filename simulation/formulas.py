        raise FormulaError(f"unknown variable: {node.id}")

    if isinstance(node, ast.UnaryOp):
        operator_fn = _UNARY_OPERATORS.get(type(node.op))
        if operator_fn is None:
            raise FormulaError("unsupported unary operator")
        return float(operator_fn(_eval_node(node.operand, names)))

    if isinstance(node, ast.BinOp):
        operator_fn = _BINARY_OPERATORS.get(type(node.op))
        if operator_fn is None:
            raise FormulaError("unsupported binary operator")
        left = _eval_node(node.left, names)
        right = _eval_node(node.right, names)
        return float(operator_fn(left, right))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise FormulaError("only simple function calls are allowed")
        function = _FUNCTIONS.get(node.func.id)
        if function is None:
            raise FormulaError(f"unknown function: {node.func.id}")
        if node.keywords:
            raise FormulaError("keyword arguments are not supported")
        args = [_eval_node(argument, names) for argument in node.args]
        return float(function(*args))

    raise FormulaError("unsupported expression")
