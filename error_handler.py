class TriageError(Exception):
    """Base class for all interpreter-related errors."""
    pass

class SyntaxTriageError(TriageError):
    """Raised when the syntax of a command is invalid."""
    def __init__(self, message):
        super().__init__(f"Syntax Error: {message}")

class RuntimeTriageError(TriageError):
    """Raised when a runtime issue occurs during rule evaluation."""
    def __init__(self, message):
        super().__init__(f"Runtime Error: {message}")
