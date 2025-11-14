"""
Main entry point for the Medical Triage Interpreter.

This file is respons        # Hospital alert thresholds (adjustable)
        if high_risk_count > 50:
            hospital_alert = "CRITICAL ALERT"
            alert_level = 2
        elif high_risk_count > 20:
            hospital_alert = "WARNING ALERT"
            alert_level = 1
        else:
            hospital_alert = "NORMAL"
            alert_level = 0
1. Reading the rule script(s) from the /tests directory.
2. Reading the patient data from the /data directory.
3. Running the full interpreter pipeline:
   - Lexer: Tokenizes the rule script.
   - Parser: Creates an Abstract Syntax Tree (AST) from tokens.
   - Executor: Loads data and executes the AST against it.
4. Printing the final triaged data to the console.

This demonstrates Section 6 (Testing) for the full system.
"""

import sys
from pathlib import Path
import pandas as pd
from lexer import Lexer
from triage_parser import Parser
from executor import Executor
from error_handler import TriageError

# --- File Paths ---
# Resolve paths relative to this file so the script works from any CWD
BASE_DIR = Path(__file__).resolve().parent
DATA_FILEPATH = BASE_DIR / 'data' / 'human_vital_sign.csv'
VALID_RULES_FILEPATH = BASE_DIR / 'tests' / 'test_valid_inputs.txt'
INVALID_RULES_FILEPATH = BASE_DIR / 'tests' / 'text_invalid_inputs.txt'

def read_file_content(filepath):
    """Helper function to read a file and return its content."""
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Fatal Error: File not found at {filepath}", file=sys.stderr)
        print("Please ensure the file exists and you are running main.py from the project's root directory.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Fatal Error: Could not read file at {filepath}. Error: {e}", file=sys.stderr)
        sys.exit(1)

def run_interpreter(rule_script, data_filepath):
    """
    Runs the full Lexer -> Parser -> Executor pipeline on a given
    rule script and data file.
    """
    print("--- Medical Triage Interpreter ---")
    print(f"Loading rules...\n{rule_script}")
    print(f"Loading data from: {data_filepath}\n")

    try:
        # 1. LEXER: Tokenize the input script
        print("[1/3] Running Lexer...")
        lexer = Lexer(rule_script)
        tokens = lexer.tokenize()
        print(f"Lexer found {len(tokens)} tokens.")
        
        # 2. PARSER: Build the Abstract Syntax Tree
        print("[2/3] Running Parser...")
        parser = Parser(tokens)
        ast = parser.parse()
        print(f"Parser created AST with {len(ast.rules)} rules.")
        
        # 3. EXECUTOR: Load data and run the rules
        print("[3/3] Running Executor...")
        executor = Executor(ast)
        executor.load_data(data_filepath) # This also runs feature engineering
        print(f"Data loaded. {len(executor.data)} patients found.")
        print("Applying rules to all patients...")
        
        results_df = executor.execute()
        
        print("\n--- Triage Execution Complete ---")
        
        print("\n--- Hospital-Level Alert Status ---")
        print(f"High Risk Patients: {results_df['High_Risk_Count'].iloc[0]:,} / {results_df['Total_Patients'].iloc[0]:,} ({(results_df['High_Risk_Count'].iloc[0] / results_df['Total_Patients'].iloc[0] * 100):.2f}%)")
        print(f"Hospital Alert: {results_df['Hospital_Alert'].iloc[0]}")
        print(f"Alert Level: {results_df['Alert_Level'].iloc[0]}")
        
        # Display the results with all columns properly expanded
        result_columns = [
            'Patient ID', 'Heart Rate', 'Oxygen Saturation', 'Derived_BMI',
            'Risk', 'Fever', 'Needs_Obesity_Consult', 'Recommendation'
        ]
        
        # Filter for columns that actually exist in the final dataframe
        display_cols = [col for col in result_columns if col in results_df.columns]
        
        # Expand pandas display to show all columns without truncation
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_rows', None)
        
        # Show first 10 and last 10 patients
        print("\n--- Patient Triage Results (Sample) ---")
        print("\nFirst 10 patients:")
        print(results_df[display_cols].head(10))
        print("\n...")
        print("\nLast 10 patients:")
        print(results_df[display_cols].tail(10))

        print(f"\nSuccessfully processed {len(results_df)} patients.")
        
    except TriageError as e:
        print(f"\n--- INTERPRETER FAILED ---")
        print(f"An Error Occured:\n{e}")
    except Exception as e:
        print(f"\n--- SYSTEM ERROR ---")
        print(f"An Error Occured:\n{e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Main function to run the tests as specified in Section 6.
    """
    
    # --- Test 1: Valid Rule Script ---
    print("=========================================")
    print("           Running Test 1: VALID         ")
    print("=========================================")
    valid_rule_script = read_file_content(VALID_RULES_FILEPATH)
    run_interpreter(valid_rule_script, DATA_FILEPATH)
    
    
    # --- Test 2: Invalid Rule Script (Error Handling) ---
    print("\n\n=========================================")
    print("          Running Test 2: INVALID        ")
    print("=========================================")
    invalid_rule_script = read_file_content(INVALID_RULES_FILEPATH)
    run_interpreter(invalid_rule_script, DATA_FILEPATH)

if __name__ == "__main__":
    main() 
