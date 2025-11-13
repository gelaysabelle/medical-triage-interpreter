from error_handler import SyntaxTriageError

def parse(tokens):
    """
    Parse the list of tokens into a rule dictionary.
    Example output:
    {
        "conditions": [
            {"metric": "heart_rate", "operator": ">", "value": 100, "connector": "AND"},
            {"metric": "oxygen_saturation", "operator": "<", "value": 95}
        ],
        "result": "CRITICAL"
    }
    """
    if not tokens:
        raise SyntaxTriageError("Empty input.")

    if tokens[0][1] != "IF":
        raise SyntaxTriageError("Missing 'IF' at start of rule.")

    conditions = []
    i = 1

    while i < len(tokens) and tokens[i][1] != "THEN":
        try:
            metric = tokens[i][1]
            op = tokens[i + 1][1]
            val = float(tokens[i + 2][1])
            cond = {"metric": metric, "operator": op, "value": val}
            i += 3
            if i < len(tokens) and tokens[i][1] in ("AND", "OR"):
                cond["connector"] = tokens[i][1]
                i += 1
            conditions.append(cond)
        except (IndexError, ValueError):
            raise SyntaxTriageError("Invalid condition structure.")

    if i >= len(tokens) - 1 or tokens[i][1] != "THEN":
        raise SyntaxTriageError("Missing 'THEN' keyword.")
    
    result = tokens[i + 1][1]
    return {"conditions": conditions, "result": result}
