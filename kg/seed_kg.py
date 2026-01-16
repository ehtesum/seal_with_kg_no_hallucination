from dynamic_kg import ensure_and_save_condition, load_graph, get_symptoms

# Seed anxiety once
ensure_and_save_condition(
    "Anxiety",
    "Anxiety",
    ["restlessness", "muscle tension", "worry"]
)

g = load_graph()
print("Anxiety symptoms in KG:", get_symptoms(g, "Anxiety"))
