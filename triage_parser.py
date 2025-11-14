"""
Parser (Syntax Analyzer) for the Medical Triage Interpreter

This module takes the list of tokens from the Lexer and builds an
Abstract Syntax Tree (AST). The AST is a tree-like structure that
represents the grammar and logic of the rule script.

Supports new keywords: COUNT (for counting high-risk patients) and ELSE.
"""

from lexer import TokenType
from error_handler import ParserError

###############################################################################
# AST (Abstract Syntax Tree) Node Classes
###############################################################################

class ASTNode:
    """Base class for all AST nodes."""
    pass

class RuleScriptNode(ASTNode):
    """Represents the entire script, containing a list of rules."""
    def __init__(self, rules):
        self.rules = rules # List of RuleNode objects

class RuleNode(ASTNode):
    """Represents a single 'IF ... THEN ... (ELSE ...)' rule."""
    def __init__(self, condition, then_actions, else_actions=None):
        self.condition = condition # A condition node
        self.then_actions = then_actions     # A list of action nodes (SetActionNode or nested RuleNode)
        self.else_actions = else_actions     # NEW: Optional list of actions for ELSE branch

class SetActionNode(ASTNode):
    """Represents a 'SET variable = value' action."""
    def __init__(self, identifier, value_node):
        self.identifier = identifier # The new/updated column name (e.g., 'Risk')
        self.value_node = value_node # A ValueNode (String, Number, etc.) or CountNode

class BinaryOpNode(ASTNode):
    """Represents a binary operation (e.g., AND, OR)."""
    def __init__(self, left, op_token, right):
        self.left = left
        self.op_token = op_token
        self.right = right

class UnaryOpNode(ASTNode):
    """Represents a unary operation (e.g., NOT)."""
    def __init__(self, op_token, operand):
        self.op_token = op_token
        self.operand = operand

class ComparisonNode(ASTNode):
    """Represents a comparison (e.g., 'Heart Rate' > 100)."""
    def __init__(self, left_identifier, op_token, right_value):
        self.left_identifier = left_identifier
        self.op_token = op_token
        self.right_value = right_value # ValueNode or IdentifierNode

class IsNullNode(ASTNode):
    """Represents a check for NULL (e.g., 'Blood Pressure' IS NULL)."""
    def __init__(self, identifier, is_not):
        self.identifier = identifier
        self.is_not = is_not # Boolean, True if "IS NOT NULL"

class CountNode(ASTNode):
    """Represents a COUNT function (e.g., COUNT WHERE 'Risk' == "High Risk")."""
    def __init__(self, condition):
        self.condition = condition # The condition to count matching rows

class ValueNode(ASTNode):
    """Represents a literal value (String, Number, Boolean, Null)."""
    def __init__(self, token):
        self.token = token
        self.value = token.value

class IdentifierNode(ASTNode):
    """Represents an identifier (e.g., a column name)."""
    def __init__(self, token):
        self.token = token
        self.name = token.value

###############################################################################
# Parser Class
###############################################################################

class Parser:
    """
    Parses a list of tokens into an Abstract Syntax Tree (AST).
    This is a recursive descent parser.
    """
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        if not self.tokens:
             raise ParserError("Received empty token list.", 1, 1)
        self.current_token = self.tokens[self.pos]

    def advance(self):
        """Move to the next token."""
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
        else:
            # This condition should not be met if EOF is handled properly
            self.current_token = None 

    def consume(self, expected_token_type):
        """
        Consume the current token if it matches the expected type,
        otherwise raise a ParserError.
        """
        if self.current_token.type == expected_token_type:
            token = self.current_token
            self.advance()
            return token
        else:
            raise ParserError(
                f"Expected {expected_token_type.name} but got {self.current_token.type.name}",
                self.current_token.line, self.current_token.col
            )

    def parse(self):
        """Main entry point. Parses the entire script."""
        rules = []
        # Skip leading newlines
        while self.current_token.type == TokenType.NEWLINE:
            self.advance()
        
        while self.current_token.type != TokenType.EOF:
            rules.append(self.parse_rule(is_nested=False))
            # Skip any newlines between rules
            while self.current_token.type == TokenType.NEWLINE:
                self.advance()
        
        if not rules:
            raise ParserError("Empty rule script. Expected one or more 'IF' rules.", 1, 1)
            
        return RuleScriptNode(rules)

    def parse_rule(self, is_nested=False):
        """
        Parses a single 'IF ... THEN ... (ELSE ...)' rule.
        If is_nested=True, this is a nested IF inside a THEN/ELSE block.
        If is_nested=False, this is a top-level rule and should NOT consume IFs after blank lines.
        """
        self.consume(TokenType.IF)
        
        condition = self.parse_condition()
        
        self.consume(TokenType.THEN)
        
        # Skip newlines after THEN
        while self.current_token.type == TokenType.NEWLINE:
            self.advance()
        
        # Parse THEN actions (can include nested IFs or SETs)
        # For top-level rules, stop if we see IF after a blank line
        then_actions = []
        while self.current_token.type in (TokenType.SET, TokenType.IF):
            if self.current_token.type == TokenType.SET:
                then_actions.append(self.parse_action())
                # Track if there's a blank line after this SET
                blank_line_after_set = False
                newline_count = 0
                while self.current_token.type == TokenType.NEWLINE:
                    newline_count += 1
                    self.advance()
                # If there were multiple newlines (blank line), mark it
                if newline_count > 1:
                    blank_line_after_set = True
                
                # If this is a top-level rule and we see IF after blank line, stop parsing THEN
                if not is_nested and blank_line_after_set and self.current_token.type == TokenType.IF:
                    break
            elif self.current_token.type == TokenType.IF:
                # Nested IF statement - always parse as nested
                then_actions.append(self.parse_rule(is_nested=True))
                # Skip newlines after nested IF
                blank_line_after_if = False
                newline_count = 0
                while self.current_token.type == TokenType.NEWLINE:
                    newline_count += 1
                    self.advance()
                if newline_count > 1:
                    blank_line_after_if = True
                
                # If this is a top-level rule and we see IF after blank line, stop parsing THEN
                if not is_nested and blank_line_after_if and self.current_token.type == TokenType.IF:
                    break
            
        if not then_actions:
            raise ParserError("Expected at least one action (SET or nested IF) after 'THEN'",
                              self.current_token.line, self.current_token.col)
        
        # Parse optional ELSE clause
        else_actions = None
        if self.current_token.type == TokenType.ELSE:
            self.advance()
            # Skip newlines after ELSE
            while self.current_token.type == TokenType.NEWLINE:
                self.advance()
            
            else_actions = []
            while self.current_token.type in (TokenType.SET, TokenType.IF):
                if self.current_token.type == TokenType.SET:
                    else_actions.append(self.parse_action())
                    # Skip newlines after SET
                    blank_line_after_set = False
                    newline_count = 0
                    while self.current_token.type == TokenType.NEWLINE:
                        newline_count += 1
                        self.advance()
                    if newline_count > 1:
                        blank_line_after_set = True
                    # Stop if we hit blank line + next rule in top-level ELSE
                    if not is_nested and blank_line_after_set and self.current_token.type == TokenType.IF:
                        break
                elif self.current_token.type == TokenType.IF:
                    # Nested IF in ELSE
                    else_actions.append(self.parse_rule(is_nested=True))
                    # Skip newlines after nested IF
                    blank_line_after_if = False
                    newline_count = 0
                    while self.current_token.type == TokenType.NEWLINE:
                        newline_count += 1
                        self.advance()
                    if newline_count > 1:
                        blank_line_after_if = True
                    # Stop if we hit blank line + next rule
                    if not is_nested and blank_line_after_if and self.current_token.type == TokenType.IF:
                        break
            
            if not else_actions:
                raise ParserError("Expected at least one action (SET or nested IF) after 'ELSE'",
                                  self.current_token.line, self.current_token.col)
                              
        return RuleNode(condition=condition, then_actions=then_actions, else_actions=else_actions)

    def parse_action(self):
        """Parses a 'SET IDENTIFIER = value' action."""
        self.consume(TokenType.SET)
        
        identifier_token = self.consume(TokenType.IDENTIFIER)
        # We use an IdentifierNode to store the *name* of the column we are setting
        identifier_node = IdentifierNode(identifier_token) 
        
        self.consume(TokenType.ASSIGN)
        
        # Check if value is a COUNT function
        if self.current_token.type == TokenType.COUNT:
            value_node = self.parse_count()
            return SetActionNode(identifier_node, value_node)
        
        value_token_types = (TokenType.STRING, TokenType.NUMBER, TokenType.BOOLEAN, TokenType.NULL)
        if self.current_token.type in value_token_types:
            value_node = ValueNode(self.current_token)
            self.advance()
            return SetActionNode(identifier_node, value_node)
        else:
            raise ParserError(f"Expected a value (String, Number, Boolean, Null, or COUNT) but got {self.current_token.type.name}",
                              self.current_token.line, self.current_token.col)

    def parse_count(self):
        """Parses a COUNT WHERE condition function."""
        self.consume(TokenType.COUNT)
        self.consume(TokenType.WHERE)
        
        condition = self.parse_condition()
        
        return CountNode(condition)

    def parse_condition(self):
        """Parses a condition, which can have AND/OR logic."""
        node = self.parse_expression()
        
        while self.current_token.type in (TokenType.AND, TokenType.OR):
            op_token = self.current_token
            self.advance()
            right = self.parse_expression()
            node = BinaryOpNode(left=node, op_token=op_token, right=right)
            
        return node

    def parse_expression(self):
        """Parses an expression, which can have a 'NOT' prefix."""
        op_token = None
        if self.current_token.type == TokenType.NOT:
            op_token = self.current_token
            self.advance()
            
        node = self.parse_atom()
        
        if op_token:
            return UnaryOpNode(op_token=op_token, operand=node)
        return node

    def parse_atom(self):
        """
        Parses the smallest part of a condition:
        - ( ... condition ... )
        - IDENTIFIER > VALUE
        - IDENTIFIER IS NULL
        """
        
        # Case 1: Parenthesized expression ( ... )
        if self.current_token.type == TokenType.LPAREN:
            self.advance()
            node = self.parse_condition()
            self.consume(TokenType.RPAREN)
            return node
            
        # Case 2: Identifier-based expression
        if self.current_token.type == TokenType.IDENTIFIER:
            identifier_node = IdentifierNode(self.current_token)
            self.advance()
            
            # Sub-case 2a: 'IDENTIFIER IS (NOT)? NULL'
            if self.current_token.type == TokenType.IS:
                self.advance()
                is_not = False
                if self.current_token.type == TokenType.NOT:
                    is_not = True
                    self.advance()
                self.consume(TokenType.NULL)
                return IsNullNode(identifier=identifier_node, is_not=is_not)
                
            # Sub-case 2b: 'IDENTIFIER <op> VALUE'
            op_types = (TokenType.GT, TokenType.LT, TokenType.GTE, TokenType.LTE, TokenType.EQ, TokenType.NEQ)
            if self.current_token.type in op_types:
                op_token = self.current_token
                self.advance()
                
                # The right side can be a literal value OR another column
                value_types = (TokenType.NUMBER, TokenType.STRING, TokenType.BOOLEAN, TokenType.IDENTIFIER)
                if self.current_token.type in value_types:
                    if self.current_token.type == TokenType.IDENTIFIER:
                        right_node = IdentifierNode(self.current_token)
                    else:
                        right_node = ValueNode(self.current_token)
                    self.advance()
                    return ComparisonNode(left_identifier=identifier_node, op_token=op_token, right_value=right_node)
                else:
                    raise ParserError(f"Expected Number, String, Boolean, or Identifier after operator",
                                      self.current_token.line, self.current_token.col)

        # If we get here, the syntax is wrong
        raise ParserError(f"Unexpected token in expression: {self.current_token.type.name}",
                          self.current_token.line, self.current_token.col)
