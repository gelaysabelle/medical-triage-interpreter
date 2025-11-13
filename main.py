from lexer import tokenize
from parser import parse
from executor import execute_rule
import pandas as pd

# Load dataset
data = pd.read_csv("data/human_vital_sign.csv")

data.columns = [c.lower().replace(" ", "_").replace("(", "").replace(")", "") for c in data.columns]


# Sample rule input
rule_input = "IF heart_rate > 30 AND oxygen_saturation < 99 THEN CRITICAL"

# Tokenize → Parse → Execute
tokens = tokenize(rule_input)
parsed_rule = parse(tokens)
print("\nParsed Rule:", parsed_rule)

result_counts = execute_rule(data, parsed_rule)
print("\nTriage Classification Summary:")
print(result_counts)
