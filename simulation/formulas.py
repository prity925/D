"""安全表达式求值引擎"""
from __future__ import annotations
import ast
import math
import operator
from collections.abc import Mapping

class FormulaError(ValueError):
    pass

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_FUNCTIONS = {
    "abs": abs,
    "ceil": math.ceil,
    "clip": lambda value, low, high: min(max(value, low), high),
    "exp": math.exp,
    "floor": math.floor,
    "log": math.log,
    "max": max,
    "min": min,
    "pow": pow,
    "round": round,
    "sqrt": math.sqrt,
}

_CONSTANTS = {
    "e": math.e,
    "pi": math.pi,
}

def clamp(value: float, minimum: float | None = None, maximum: float | None = None) -> float:
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value

def evaluate_expression(
    expression: str,
    *,
    variables: Mapping[str, float] | None = None,
    default: float | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    text = expression.strip()
    if not text:
        if default is None:
            raise FormulaError("empty expression")
        return clamp(float(default), minimum, maximum)

    names = dict(_CONSTANTS)
    if variables:
        names.update({name: float(value) for name, value in variables.items()})

    try:
        tree = ast.parse(text, mode="eval")
        value = _eval_node(tree.body, names)
    except FormulaError:
        raise
    except Exception as exc:
        raise FormulaError(str(exc)) from exc

    if not math.isfinite(value):
        raise FormulaError("expression did not produce a finite number")
    return clamp(float(value), minimum, maximum)

def _eval_node(node: ast.AST, names: Mapping[str, float]) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int | float):
            return float(node.value)
        raise FormulaError("only numeric constants are allowed")

    if isinstance(node, ast.Name):
        if node.id in names:
            return float(names[node.id])
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
