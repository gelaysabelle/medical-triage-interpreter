import pandas as pd
from error_handler import RuntimeTriageError

def evaluate_condition(row, cond):
    lhs = row.get(cond["metric"])
    rhs = cond["value"]
    op = cond["operator"]

    if lhs is None:
        raise RuntimeTriageError(f"Unknown metric: {cond['metric']}")

    if op == ">": return lhs > rhs
    if op == "<": return lhs < rhs
    if op == ">=": return lhs >= rhs
    if op == "<=": return lhs <= rhs
    if op == "==": return lhs == rhs
    if op == "!=": return lhs != rhs
    raise RuntimeTriageError(f"Invalid operator: {op}")

def execute_rule(data, rule):
    results = []
    for _, row in data.iterrows():
        truth = True
        for cond in rule["conditions"]:
            res = evaluate_condition(row, cond)
            if "connector" in cond:
                if cond["connector"] == "AND":
                    truth = truth and res
                elif cond["connector"] == "OR":
                    truth = truth or res
            else:
                truth = res
        results.append(rule["result"] if truth else "NORMAL")

    data["Triage"] = results
    return data["Triage"].value_counts()
