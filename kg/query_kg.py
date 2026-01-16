from rdflib import Graph

def load_kg(path):
    g = Graph()
    g.parse(path, format="turtle")
    return g

def get_symptoms_of_anxiety(g):
    query = """
    PREFIX mh: <http://example.org/mentalhealth#>

    SELECT ?symptomLabel WHERE {
        mh:Anxiety mh:associated_with ?symptom .
        ?symptom mh:label ?symptomLabel .
    }
    """
    results = g.query(query)
    return [str(row[0]) for row in results]

def get_symptoms_of_depression(g):
    query = """
    PREFIX mh: <http://example.org/mentalhealth#>

    SELECT ?symptomLabel WHERE {
        mh:Depression mh:associated_with ?symptom .
        ?symptom mh:label ?symptomLabel .
    }
    """
    results = g.query(query)
    return [str(row[0]) for row in results]


def get_symptoms_of_ocd(g):
    query = """
    PREFIX mh: <http://example.org/mentalhealth#>

    SELECT ?symptomLabel WHERE {
        mh:OCD mh:associated_with ?symptom .
        ?symptom mh:label ?symptomLabel .
    }
    """
    results = g.query(query)
    return [str(row[0]) for row in results]

def get_symptoms_of_schizophrenia(g):
    query = """
    PREFIX mh: <http://example.org/mentalhealth#>

    SELECT ?symptomLabel WHERE {
        mh:Schizophrenia mh:associated_with ?symptom .
        ?symptom mh:label ?symptomLabel .
    }
    """
    results = g.query(query)
    return [str(row[0]) for row in results]


