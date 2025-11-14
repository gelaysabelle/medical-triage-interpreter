"""
Executor (Interpreter Engine) for the Medical Triage Interpreter

This module takes the Abstract Syntax Tree (AST) from the Parser
and executes the logic against a patient dataset.

It uses the Visitor design pattern to traverse the AST.

Supports COUNT() and ELSE blocks in rules.
Generates hospital-level status separately from patient rows.
"""

import pandas as pd
import numpy as np
from error_handler import ExecutorError
from triage_parser import (
    ASTNode, RuleScriptNode, RuleNode, SetActionNode, BinaryOpNode,
    UnaryOpNode, ComparisonNode, IsNullNode, ValueNode, IdentifierNode,
    CountNode
)
from lexer import TokenType

class Executor:
    """
    Executes the parsed rule script (AST) on a loaded DataFrame.
    """
    def __init__(self, ast):
        if not isinstance(ast, RuleScriptNode):
            raise ExecutorError("Executor must be initialized with a valid RuleScriptNode.")
        self.ast = ast
        self.data = None
        self.all_rows = None  # For COUNT function access
        self.count_cache = {}  # Cache COUNT results per rule iteration

    def load_data(self, filepath):
        """
        Loads patient data from a CSV file and performs feature engineering
        as defined in the Data_Preprocessing_&_Cleaning.ipynb notebook.
        """
        try:
            self.data = pd.read_csv(filepath)
            print(f"Successfully loaded '{filepath}'.")
        except FileNotFoundError:
            raise ExecutorError(f"Data file not found at path: {filepath}")
        except Exception as e:
            raise ExecutorError(f"Error loading data: {e}")
            
        # Perform Feature Engineering, replicating the notebook's logic.
        # This makes 'Pulse_Pressure', 'BMI', and 'MAP' available to the rule language.
        try:
            print("Performing feature engineering...")
            # Ensure columns exist before calculation
            required_cols = ['Systolic Blood Pressure', 'Diastolic Blood Pressure', 'Weight (kg)', 'Height (m)']
            for col in required_cols:
                if col not in self.data.columns:
                    print(f"Warning: Missing data column for feature engineering: '{col}'. Rules using this column may fail.")
            
            # Handle division by zero for BMI by setting non-positive heights to NaN (vectorized)
            if 'Height (m)' in self.data.columns:
                self.data['Height (m)'] = self.data['Height (m)'].where(self.data['Height (m)'] > 0, np.nan)
            
            # Calculate engineered features
            if 'Systolic Blood Pressure' in self.data.columns and 'Diastolic Blood Pressure' in self.data.columns:
                self.data['Pulse_Pressure'] = self.data['Systolic Blood Pressure'] - self.data['Diastolic Blood Pressure']
                self.data['MAP'] = self.data['Diastolic Blood Pressure'] + (self.data['Pulse_Pressure'] / 3)
            
            if 'Weight (kg)' in self.data.columns and 'Height (m)' in self.data.columns:
                self.data['BMI'] = self.data['Weight (kg)'] / (self.data['Height (m)'] ** 2)
            
            # Replace infinities from division by zero with NaN (which 'IS NULL' can catch)
            self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
            print("Feature engineering complete.")
            
        except Exception as e:
            raise ExecutorError(f"Error during feature engineering: {e}")

    def execute(self):
        """
        Executes the full rule script on the loaded data.
        It iterates through each rule, and for each rule, iterates
        through each patient (row).
        """
        if self.data is None:
            raise ExecutorError("Data not loaded. Call load_data(filepath) first.")
            
        # We create a dictionary from the dataframe for faster row-by-row processing
        # and to allow setting new values.
        results = self.data.to_dict('records')
        self.all_rows = results  # Store for COUNT function
        
        for rule_node in self.ast.rules:
            # Reset COUNT cache for each rule to ensure fresh evaluations
            self.count_cache = {}
            
            for row in results:
                # The 'row' is our context for this execution
                try:
                    self.visit(rule_node, row)
                except Exception as e:
                    # Catch runtime errors during rule evaluation (e.g., comparing string to number)
                    # We will log this error and continue to the next row
                    patient_id = row.get('Patient ID', 'Unknown')
                    print(f"Runtime Warning: {e}. Skipping rule for row {patient_id}")

        # Convert the list of dictionaries back to a DataFrame
        return pd.DataFrame(results)


    def visit(self, node, context_row):
        """
        Generic visit method that dispatches to the correct node-specific visitor.
        'context_row' is the current patient data (a dictionary).
        """
        method_name = f'visit_{type(node).__name__}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node, context_row)

    def generic_visit(self, node, context_row):
        """Called if no explicit visitor method exists."""
        raise ExecutorError(f"No visitor method found for AST node type: {type(node).__name__}")

    def visit_RuleNode(self, node, context_row):
        """Visits a rule node with condition, then_actions, and optional else_actions."""
        condition_result = self.visit(node.condition, context_row)
        
        if condition_result:
            # Execute THEN branch
            for action_node in node.then_actions:
                self.visit(action_node, context_row)
        elif node.else_actions is not None:
            # Execute ELSE branch if condition is false
            for action_node in node.else_actions:
                self.visit(action_node, context_row)
        
        return True

    def visit_BinaryOpNode(self, node, context_row):
        """Visits AND/OR nodes with short-circuit evaluation."""
        # Short-circuit AND: if left is false, don't evaluate right
        if node.op_token.type == TokenType.AND:
            left_val = self.visit(node.left, context_row)
            if not left_val:
                return False
            right_val = self.visit(node.right, context_row)
            return bool(left_val and right_val)
        
        # Short-circuit OR: if left is true, don't evaluate right
        elif node.op_token.type == TokenType.OR:
            left_val = self.visit(node.left, context_row)
            if left_val:
                return True
            right_val = self.visit(node.right, context_row)
            return bool(left_val or right_val)

    def visit_UnaryOpNode(self, node, context_row):
        """Visits NOT nodes."""
        operand_val = self.visit(node.operand, context_row)
        if node.op_token.type == TokenType.NOT:
            return not bool(operand_val)
            
    def visit_ComparisonNode(self, node, context_row):
        """Visits comparison nodes (e.g., 'Heart Rate' > 100)."""
        
        # Get the value from the patient data
        left_val = self.visit(node.left_identifier, context_row)
        
        # Get the value to compare against (could be a literal or another column)
        right_val = self.visit(node.right_value, context_row)
        
        # Handle NULLs: any comparison with a NULL value is False
        if pd.isna(left_val) or pd.isna(right_val):
            return False
            
        # Perform the comparison
        op = node.op_token.type
        try:
            if op == TokenType.GT: return left_val > right_val
            if op == TokenType.LT: return left_val < right_val
            if op == TokenType.GTE: return left_val >= right_val
            if op == TokenType.LTE: return left_val <= right_val
            if op == TokenType.EQ: return left_val == right_val
            if op == TokenType.NEQ: return left_val != right_val
        except TypeError as e:
            # Catch type errors (e.g., 'Age' > "High")
            raise ExecutorError(f"Type mismatch during comparison: {left_val} vs {right_val}. Error: {e}")

    def visit_IsNullNode(self, node, context_row):
        """Visits 'IS NULL' or 'IS NOT NULL' nodes."""
        value = self.visit(node.identifier, context_row)
        
        is_null = pd.isna(value)
        
        return not is_null if node.is_not else is_null

    def visit_CountNode(self, node, context_row):
        """
        Visits a COUNT node. Counts how many rows in the entire dataset
        match the given condition. Uses caching to avoid re-computing
        the same condition multiple times per rule.
        """
        # Create a cache key from the condition node's representation
        condition_key = id(node.condition)
        
        # Check if we've already computed this count in this rule iteration
        if condition_key in self.count_cache:
            return self.count_cache[condition_key]
        
        count = 0
        for row in self.all_rows:
            try:
                if self.visit(node.condition, row):
                    count += 1
            except Exception:
                # Skip rows that cause errors in the count condition
                pass
        
        # Cache the result for this rule iteration
        self.count_cache[condition_key] = count
        return count

    def visit_ValueNode(self, node, context_row):
        """Returns the raw literal value (String, Number, etc.)."""
        # Special case for NULL token
        if node.token.type == TokenType.NULL:
            return np.nan # Use numpy's NaN for our internal NULL representation
        return node.value

    def visit_IdentifierNode(self, node, context_row):
        """Gets a value from the patient data row (context)."""
        if node.name not in context_row:
            # This allows rules to SET a new variable and read it later
            # (e.g., SET Fever = True ... IF Fever == True THEN ...)
            context_row[node.name] = np.nan 
        
        return context_row.get(node.name)

    def visit_SetActionNode(self, node, context_row):
        """Executes a 'SET' action, modifying the patient row."""
        new_value = self.visit(node.value_node, context_row)
        column_to_set = node.identifier.name
        
        # Modify the row in-place
        context_row[column_to_set] = new_value
        return True