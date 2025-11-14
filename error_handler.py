"""
Error Handler for the Medical Triage Interpreter

This file defines the custom exception classes that are raised by
the Lexer, Parser, and Executor to provide clear, actionable error
messages.

This fulfills Section 5 (Implementation) for Error Handling.
"""

class TriageError(Exception):
    """Base class for all errors in the interpreter."""
    def __init__(self, message, line_num=1, col_num=1):
        super().__init__(message)
        self.message = message
        self.line_num = line_num
        self.col_num = col_num

    def __str__(self):
        return f"[Line {self.line_num}:{self.col_num}] {type(self).__name__}: {self.message}"


class LexerError(TriageError):
    """Raised when the lexer encounters an invalid character or malformed token."""
    pass


class ParserError(TriageError):
    """Raised when the parser encounters a syntax error."""
    pass


class ExecutorError(TriageError):
    """Raised during rule execution (e.g., type mismatch, missing variable)."""
    def __init__(self, message, line_num=None, col_num=None):
        super().__init__(message)
        self.message = message
        self.line_num = line_num
        self.col_num = col_num
    
    def __str__(self):
        if self.line_num:
            return f"[Line {self.line_num}:{self.col_num}] {type(self).__name__}: {self.message}"
        return f"{type(self).__name__}: {self.message}"
