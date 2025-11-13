"""
Main entry point for the Medical Triage Interpreter.

This file is responsible for:
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
        
        # Display the results
        # We show only the key columns for brevity
        result_columns = [
            'Patient ID', 'Heart Rate', 'Oxygen Saturation', 'BMI',
            'Risk', 'Fever', 'Needs_Obesity_Consult'
        ]
        
        # Filter for columns that actually exist in the final dataframe
        display_cols = [col for col in result_columns if col in results_df.columns]
        
        # Show first 10 and last 10 patients
        print("--- Triage Results (Sample) ---")
        with pd.option_context('display.max_rows', 20, 'display.width', 1000):
            print(results_df[display_cols].head(10))
            print("...")
            print(results_df[display_cols].tail(10))

        print(f"\nSuccessfully processed {len(results_df)} patients.")
        
    except TriageError as e:
        print(f"\n--- INTERPRETER FAILED ---")
        print(f"An error occurred:\n{e}", file=sys.stderr)
    except Exception as e:
        print(f"\n--- SYSTEM ERROR ---")
        print(f"An unexpected Python error occurred:\n{e}", file=sys.stderr)

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

#DATA PREPROCESSING 

import pandas as pd

# Load the dataset
file_name = "human_vital_signs_dataset_2024.csv"
df = pd.read_csv(file_name)

# Display the first few rows
print("--- DataFrame Head ---")
print(df.head().to_markdown(index=False, numalign="left", stralign="left"))

# Display column information
print("\n--- DataFrame Info ---")
df.info()


# Convert 'Timestamp' column to datetime objects
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Verify the conversion
print(df['Timestamp'].dtype)


# Check unique values in 'Gender' and 'Risk Category'
print("\nUnique values in 'Gender':", df['Gender'].unique())
print("Unique values in 'Risk Category':", df['Risk Category'].unique())

# Assuming we find values like ['Male', 'Female', 'male', 'female'], we would standardize them:
# df['Gender'] = df['Gender'].str.capitalize()



# Example for Heart Rate outlier detection (assuming a plausible range)
hr_min = 40
hr_max = 180
outliers_hr = df[(df['Heart Rate'] < hr_min) | (df['Heart Rate'] > hr_max)]
print(f"\nNumber of Heart Rate outliers (outside {hr_min}-{hr_max} bpm): {len(outliers_hr)}")

# Handling: Removing the identified outliers (if chosen)
# df = df[(df['Heart Rate'] >= hr_min) & (df['Heart Rate'] <= hr_max)]


# Extract the hour of the reading
df['Reading_Hour'] = df['Timestamp'].dt.hour

# Extract the day of the week
df['Day_of_Week'] = df['Timestamp'].dt.day_name()

# Display the new columns
print("\nDataFrame with new features:")
print(df[['Timestamp', 'Reading_Hour', 'Day_of_Week']].head().to_markdown(index=False, numalign="left", stralign="left"))



# 1. Label Encode the 'Risk Category' (Target)
# Assuming 'High Risk' = 1 and 'Low Risk' = 0
df['Risk Category Encoded'] = df['Risk Category'].map({'High Risk': 1, 'Low Risk': 0})

# 2. Label Encode 'Gender'
# Assuming 'Male' = 1 and 'Female' = 0 (or vice versa, the order doesn't drastically change for binary)
df['Gender Encoded'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Drop the original categorical columns
df = df.drop(columns=['Gender', 'Risk Category'])

print("\n--- After Encoding and Dropping Originals ---")
print(df[['Gender Encoded', 'Risk Category Encoded']].head().to_markdown(index=False, numalign="left", stralign="left"))



from sklearn.preprocessing import StandardScaler
import numpy as np

# Identify numerical columns for scaling (excluding IDs and the encoded/datetime columns)
# We will use all float64 and int64 columns except Patient ID, and the encoded columns
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
cols_to_exclude = ['Patient ID', 'Gender Encoded', 'Risk Category Encoded']
features_to_scale = [col for col in numerical_cols if col not in cols_to_exclude]

# Initialize the Scaler
scaler = StandardScaler()

# Apply Standardization to the selected features
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

print("\n--- After Feature Scaling (Standardization) ---")
print(df[features_to_scale].head().to_markdown(index=False, numalign="left", stralign="left"))



# Check the correlation matrix for highly correlated features
# Example: Correlation between derived features and their source features
correlation_check = df[['Systolic Blood Pressure', 'Diastolic Blood Pressure', 'Derived_Pulse_Pressure', 'Derived_MAP']].corr()

print("\n--- Correlation Check (SBP, DBP, PP, MAP) ---")
print(correlation_check.to_markdown(numalign="left", stralign="left"))

# Action to take (if correlation is very high, e.g., > 0.95):
# Drop one of the highly correlated columns, e.g., df = df.drop(columns=['Derived_Pulse_Pressure'])



# Final check of the data types
df.info()

# Save the cleaned data to a new CSV file
# df.to_csv('human_vital_signs_cleaned.csv', index=False)


