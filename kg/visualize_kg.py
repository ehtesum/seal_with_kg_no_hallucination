from rdflib import Graph
from pyvis.network import Network
from pathlib import Path

KG_FILE = Path("knowledge_graph/mental_kg_2025-11-23_20-57-16.ttl")
OUTPUT_HTML = Path("kg_visualization.html")

def main():
    g = Graph()
    g.parse(str(KG_FILE), format="turtle")

    net = Network(height="600px", width="100%", directed=True)
    net.toggle_physics(True)

    # Simple prefix stripping for readability
    def label(node):
        s = str(node)
        if "#" in s:
            return s.split("#")[-1]
        return s

    # Add nodes + edges
    for s, p, o in g:
        s_label = label(s)
        o_label = label(o)
        p_label = label(p)

        net.add_node(s_label, label=s_label, color="#4682B4")  # blue
        net.add_node(o_label, label=o_label, color="#90EE90")  # green
        net.add_edge(s_label, o_label, title=p_label)

    # ✅ Write HTML instead of show()
    net.write_html(str(OUTPUT_HTML))

    print(f"\n✅ Interactive KG graph created!")
    print(f"📁 File location: {OUTPUT_HTML}")
    print("🌐 Open it in your browser by double-clicking the file.\n")

if __name__ == "__main__":
    main()
