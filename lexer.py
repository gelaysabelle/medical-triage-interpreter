"""
Lexer (Tokenizer) for the Medical Triage Interpreter

This module reads the raw rule script as a string and breaks it down
into a list of tokens (e.g., KEYWORD, IDENTIFIER, OPERATOR).

This fulfills Section 5 (Implementation) for the Lexer.
"""

import re
from enum import Enum
from error_handler import LexerError

# Define all possible token types our language supports.
# This aligns with Section 2 (Input Language)
class TokenType(Enum):
    # Keywords
    IF = 'IF'
    THEN = 'THEN'
    ELSE = 'ELSE'  # NEW: ELSE keyword
    SET = 'SET'
    AND = 'AND'
    OR = 'OR'
    NOT = 'NOT'
    IS = 'IS'
    NULL = 'NULL'
    COUNT = 'COUNT'  # NEW: COUNT function
    WHERE = 'WHERE'  # NEW: WHERE clause for COUNT
    
    # Data Types / Values
    NUMBER = 'NUMBER'     # 100, 98.6
    STRING = 'STRING'     # "Critical", "Male"
    BOOLEAN = 'BOOLEAN'   # True, False
    IDENTIFIER = 'IDENTIFIER' # 'Heart Rate', 'Age' (using strings for multi-word)
    
    # Operators
    GT = '>'
    LT = '<'
    GTE = '>='
    LTE = '<='
    EQ = '==' # Equality check
    NEQ = '!=' # Not equal
    ASSIGN = '=' # Assignment (for SET)
    
    # Punctuation
    LPAREN = '('
    RPAREN = ')'
    NEWLINE = '\n'
    
    # End of File
    EOF = 'EOF'

class Token:
    """A simple data class to represent a single token."""
    def __init__(self, type, value, line=1, col=1):
        self.type = type
        self.value = value
        self.line = line
        self.col = col
        
    def __repr__(self):
        """String representation for easy debugging."""
        return f"Token({self.type.name}, {self.value!r}, L{self.line}:C{self.col})"

# Map of reserved keywords to their TokenTypes
RESERVED_KEYWORDS = {
    'IF': TokenType.IF,
    'THEN': TokenType.THEN,
    'ELSE': TokenType.ELSE,  # NEW
    'SET': TokenType.SET,
    'AND': TokenType.AND,
    'OR': TokenType.OR,
    'NOT': TokenType.NOT,
    'IS': TokenType.IS,
    'NULL': TokenType.NULL,
    'TRUE': TokenType.BOOLEAN,
    'FALSE': TokenType.BOOLEAN,
    'COUNT': TokenType.COUNT,  # NEW
    'WHERE': TokenType.WHERE,  # NEW
}

class Lexer:
    """
    The Lexer class, responsible for tokenizing the input string.
    """
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.current_char = self.text[self.pos] if self.pos < len(self.text) else None
        # Track position for error reporting
        self.line = 1
        self.col = 1
        
    def advance(self):
        """Move the 'pos' pointer and update 'current_char'."""
        if self.current_char == '\n':
            self.line += 1
            self.col = 0
            
        self.pos += 1
        if self.pos < len(self.text):
            self.current_char = self.text[self.pos]
            self.col += 1
        else:
            self.current_char = None

    def skip_whitespace(self):
        """Skip over spaces, tabs, and comments (but NOT newlines)."""
        while self.current_char is not None:
            # Skip spaces and tabs only; preserve newlines as tokens
            if self.current_char in (' ', '\t', '\r'):
                self.advance()
            # Also add comment skipping (skip until newline but don't consume it)
            elif self.current_char == '#':
                while self.current_char is not None and self.current_char != '\n':
                    self.advance()
            else:
                break
            
    def get_number(self):
        """Parse a number (integer or float)."""
        result = ''
        start_col = self.col
        while self.current_char is not None and (self.current_char.isdigit() or self.current_char == '.'):
            result += self.current_char
            self.advance()
            
        if result.count('.') > 1:
            raise LexerError(f"Invalid number: {result}", self.line, start_col)
            
        if '.' in result:
            return Token(TokenType.NUMBER, float(result), self.line, start_col)
        else:
            return Token(TokenType.NUMBER, int(result), self.line, start_col)

    def get_string(self, quote_char):
        """Parse a string literal (e.g., "Critical" or 'Heart Rate')."""
        result = ''
        start_col = self.col
        self.advance() # Skip opening quote
        
        while self.current_char is not None and self.current_char != quote_char:
            result += self.current_char
            self.advance()
            
        if self.current_char is None:
            raise LexerError(f"Unterminated string", self.line, start_col)
            
        self.advance() # Skip closing quote
        
        # Determine if it's a quoted identifier or a simple string value
        token_type = TokenType.IDENTIFIER if quote_char == "'" else TokenType.STRING
        return Token(token_type, result, self.line, start_col)

    def get_id_or_keyword(self):
        """Parse an identifier or a reserved keyword."""
        result = ''
        start_col = self.col
        
        # Simple identifiers (no spaces)
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
            result += self.current_char
            self.advance()

        # Check if it's a reserved keyword
        token_type = RESERVED_KEYWORDS.get(result.upper(), TokenType.IDENTIFIER)
        
        # Convert boolean strings to actual booleans
        if token_type == TokenType.BOOLEAN:
            value = True if result.upper() == 'TRUE' else False
            return Token(token_type, value, self.line, start_col)
        
        # Disallow unquoted identifiers for simplicity, except for keywords
        if token_type == TokenType.IDENTIFIER:
             raise LexerError(f"Invalid or unquoted identifier: '{result}'. Use single quotes for column names (e.g., 'Heart Rate').", self.line, start_col)

        return Token(token_type, result.upper(), self.line, start_col)

    def get_next_token(self):
        """Get the next token from the input string."""
        while self.current_char is not None:
            start_col = self.col
            
            # Emit NEWLINE tokens instead of skipping them
            if self.current_char == '\n':
                token = Token(TokenType.NEWLINE, '\n', self.line, self.col)
                self.advance()
                return token
            
            if self.current_char in (' ', '\t', '\r') or self.current_char == '#':
                self.skip_whitespace()
                continue
                
            if self.current_char.isdigit():
                return self.get_number()

            # String literal (for values)
            if self.current_char == '"':
                return self.get_string('"')
            
            # Quoted Identifier (for column names)
            if self.current_char == "'":
                return self.get_string("'")
                
            # Keywords or simple IDs
            if self.current_char.isalpha():
                return self.get_id_or_keyword()

            # Operators
            if self.current_char == '>':
                self.advance()
                if self.current_char == '=':
                    self.advance()
                    return Token(TokenType.GTE, '>=', self.line, start_col)
                return Token(TokenType.GT, '>', self.line, start_col)
                
            if self.current_char == '<':
                self.advance()
                if self.current_char == '=':
                    self.advance()
                    return Token(TokenType.LTE, '<=', self.line, start_col)
                return Token(TokenType.LT, '<', self.line, start_col)
                
            if self.current_char == '=':
                self.advance()
                if self.current_char == '=':
                    self.advance()
                    return Token(TokenType.EQ, '==', self.line, start_col)
                return Token(TokenType.ASSIGN, '=', self.line, start_col)
                
            if self.current_char == '!':
                self.advance()
                if self.current_char == '=':
                    self.advance()
                    return Token(TokenType.NEQ, '!=', self.line, start_col)
                else:
                    raise LexerError(f"Invalid character '!' (did you mean '!=')?", self.line, start_col)
            
            # Punctuation
            if self.current_char == '(':
                self.advance()
                return Token(TokenType.LPAREN, '(', self.line, start_col)
            if self.current_char == ')':
                self.advance()
                return Token(TokenType.RPAREN, ')', self.line, start_col)
                
            # If we get here, the character is invalid
            char = self.current_char
            self.advance()
            raise LexerError(f"Invalid character: '{char}'", self.line, start_col)

        # End of file
        return Token(TokenType.EOF, None, self.line, self.col)

    def tokenize(self):
        """Return a list of all tokens from the input text."""
        tokens = []
        token = self.get_next_token()
        while token.type != TokenType.EOF:
            tokens.append(token)
            token = self.get_next_token()
        tokens.append(token) # Add EOF
        return tokens