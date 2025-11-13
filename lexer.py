import re
from error_handler import SyntaxTriageError

TOKEN_SPEC = [
    ("NUMBER", r"\d+(\.\d+)?"),
    ("KEYWORD", r"\b(IF|THEN|ELSE|AND|OR)\b"),
    ("OPERATOR", r"[<>]=?|==|!="),
    ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_]*"),
    ("CATEGORY", r"\b(CRITICAL|MODERATE|NORMAL)\b"),
    ("SKIP", r"[ \t]+"),
    ("MISMATCH", r".")
]

def tokenize(code):
    """Convert a string command into a list of tokens."""
    tokens = []
    pattern = "|".join(f"(?P<{name}>{regex})" for name, regex in TOKEN_SPEC)
    for match in re.finditer(pattern, code):
        kind = match.lastgroup
        value = match.group()
        if kind == "SKIP":
            continue
        elif kind == "MISMATCH":
            raise SyntaxTriageError(f"Unexpected character: {value}")
        tokens.append((kind, value))
    return tokens
