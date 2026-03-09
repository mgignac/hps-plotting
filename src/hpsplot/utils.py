"""Safe expression parser and evaluator for selections and variable expressions.

Supports branch names, arithmetic, comparisons, boolean logic, and functions
without using eval().
"""

import re
import numpy as np

# Token types
TOKEN_NUMBER = "NUMBER"
TOKEN_NAME = "NAME"
TOKEN_OP = "OP"
TOKEN_LPAREN = "LPAREN"
TOKEN_RPAREN = "RPAREN"
TOKEN_COMMA = "COMMA"
TOKEN_EOF = "EOF"

# Supported functions
FUNCTIONS = {
    "abs": np.abs,
    "sqrt": np.sqrt,
    "log": np.log,
    "log10": np.log10,
    "exp": np.exp,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "atan2": np.arctan2,
    "arctan2": np.arctan2,
    "max": np.maximum,
    "min": np.minimum,
    "pow": np.power,
}

# Boolean keywords mapped to lowercase
BOOLEAN_KEYWORDS = {"and", "or", "not"}

# Token regex pattern
_TOKEN_RE = re.compile(
    r"""
    (?P<NUMBER>   \d+\.?\d*(?:[eE][+-]?\d+)? )  |
    (?P<NAME>     [a-zA-Z_][a-zA-Z0-9_.]* )      |
    (?P<OP>       <=|>=|!=|==|<|>|\+|-|\*\*|\*|/ )|
    (?P<LOGIC>    &&|\|\||&|\||!)                  |
    (?P<LPAREN>   \( )                            |
    (?P<RPAREN>   \) )                            |
    (?P<COMMA>    , )                             |
    (?P<WS>       \s+ )
    """,
    re.VERBOSE,
)


def tokenize(expr):
    """Tokenize an expression string into (type, value) pairs."""
    tokens = []
    for m in _TOKEN_RE.finditer(expr):
        kind = m.lastgroup
        value = m.group()
        if kind == "WS":
            continue
        if kind == "NUMBER":
            tokens.append((TOKEN_NUMBER, value))
        elif kind == "NAME":
            tokens.append((TOKEN_NAME, value))
        elif kind == "OP":
            tokens.append((TOKEN_OP, value))
        elif kind == "LOGIC":
            # Normalize C-style operators to Python-style keywords
            if value in ("&", "&&"):
                tokens.append((TOKEN_NAME, "and"))
            elif value in ("|", "||"):
                tokens.append((TOKEN_NAME, "or"))
            elif value == "!":
                tokens.append((TOKEN_NAME, "not"))
        elif kind == "LPAREN":
            tokens.append((TOKEN_LPAREN, value))
        elif kind == "RPAREN":
            tokens.append((TOKEN_RPAREN, value))
        elif kind == "COMMA":
            tokens.append((TOKEN_COMMA, value))
    tokens.append((TOKEN_EOF, ""))
    return tokens


def extract_branch_names(expr):
    """Extract branch names (identifiers that are not functions or keywords) from an expression."""
    tokens = tokenize(expr)
    branches = set()
    for i, (ttype, value) in enumerate(tokens):
        if ttype != TOKEN_NAME:
            continue
        if value.lower() in BOOLEAN_KEYWORDS:
            continue
        if value in FUNCTIONS:
            # Only skip if followed by '(' — otherwise it's a branch name
            if i + 1 < len(tokens) and tokens[i + 1][0] == TOKEN_LPAREN:
                continue
        branches.add(value)
    return branches


class Parser:
    """Recursive descent parser for safe expression evaluation.

    Grammar:
        expr       → or_expr
        or_expr    → and_expr ('or' and_expr)*
        and_expr   → not_expr ('and' not_expr)*
        not_expr   → 'not' not_expr | comparison
        comparison → addition (('>'|'<'|'>='|'<='|'=='|'!=') addition)?
        addition   → multiply (('+'|'-') multiply)*
        multiply   → power (('*'|'/') power)*
        power      → unary ('**' power)?
        unary      → '-' unary | atom
        atom       → NUMBER | NAME | NAME '(' args ')' | '(' expr ')'
    """

    def __init__(self, tokens, data):
        self.tokens = tokens
        self.pos = 0
        self.data = data

    def peek(self):
        return self.tokens[self.pos]

    def advance(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, ttype, value=None):
        tok = self.advance()
        if tok[0] != ttype or (value is not None and tok[1] != value):
            raise ValueError(f"Expected ({ttype}, {value}), got {tok}")
        return tok

    def parse(self):
        result = self.expr()
        if self.peek()[0] != TOKEN_EOF:
            raise ValueError(f"Unexpected token: {self.peek()}")
        return result

    def expr(self):
        return self.or_expr()

    def or_expr(self):
        left = self.and_expr()
        while self.peek() == (TOKEN_NAME, "or"):
            self.advance()
            right = self.and_expr()
            left = np.logical_or(left, right)
        return left

    def and_expr(self):
        left = self.not_expr()
        while self.peek() == (TOKEN_NAME, "and"):
            self.advance()
            right = self.not_expr()
            left = np.logical_and(left, right)
        return left

    def not_expr(self):
        if self.peek() == (TOKEN_NAME, "not"):
            self.advance()
            operand = self.not_expr()
            return np.logical_not(operand)
        return self.comparison()

    def comparison(self):
        left = self.addition()
        tok = self.peek()
        if tok[0] == TOKEN_OP and tok[1] in (">", "<", ">=", "<=", "==", "!="):
            op = self.advance()[1]
            right = self.addition()
            if op == ">":
                return left > right
            elif op == "<":
                return left < right
            elif op == ">=":
                return left >= right
            elif op == "<=":
                return left <= right
            elif op == "==":
                return left == right
            elif op == "!=":
                return left != right
        return left

    def addition(self):
        left = self.multiply()
        while self.peek()[0] == TOKEN_OP and self.peek()[1] in ("+", "-"):
            op = self.advance()[1]
            right = self.multiply()
            if op == "+":
                left = left + right
            else:
                left = left - right
        return left

    def multiply(self):
        left = self.power()
        while self.peek()[0] == TOKEN_OP and self.peek()[1] in ("*", "/"):
            op = self.advance()[1]
            right = self.power()
            if op == "*":
                left = left * right
            else:
                left = left / right
        return left

    def power(self):
        base = self.unary()
        if self.peek() == (TOKEN_OP, "**"):
            self.advance()
            exp = self.power()  # right-associative
            return base ** exp
        return base

    def unary(self):
        if self.peek() == (TOKEN_OP, "-"):
            self.advance()
            return -self.unary()
        return self.atom()

    def atom(self):
        tok = self.peek()

        # Number literal
        if tok[0] == TOKEN_NUMBER:
            self.advance()
            return float(tok[1])

        # Name: could be branch, function call, or boolean keyword
        if tok[0] == TOKEN_NAME:
            name = tok[1]

            # Function call
            if name in FUNCTIONS and self.pos + 1 < len(self.tokens) and \
               self.tokens[self.pos + 1][0] == TOKEN_LPAREN:
                self.advance()  # consume name
                self.expect(TOKEN_LPAREN)
                args = [self.expr()]
                while self.peek()[0] == TOKEN_COMMA:
                    self.advance()
                    args.append(self.expr())
                self.expect(TOKEN_RPAREN)
                return FUNCTIONS[name](*args)

            # Branch name
            self.advance()
            if name in self.data:
                return np.asarray(self.data[name])
            raise ValueError(f"Unknown branch: {name}")

        # Parenthesized expression
        if tok[0] == TOKEN_LPAREN:
            self.advance()
            result = self.expr()
            self.expect(TOKEN_RPAREN)
            return result

        raise ValueError(f"Unexpected token: {tok}")


def safe_evaluate(expr, data, mask=None):
    """Evaluate an expression safely against data arrays.

    Parameters
    ----------
    expr : str
        Expression string (e.g., "ele_p + pos_p", "ele_p > 0.5 and vtx_chi2 < 10").
    data : dict
        Mapping of branch names to arrays.
    mask : array-like, optional
        Boolean mask to apply to data before evaluation.

    Returns
    -------
    numpy.ndarray
        Result of evaluating the expression.
    """
    # If mask is provided, slice all data arrays
    if mask is not None:
        data = {k: np.asarray(v)[mask] for k, v in data.items()}

    tokens = tokenize(expr)
    parser = Parser(tokens, data)
    return parser.parse()
