import sys
import json
import pickle
from pathlib import Path
from typing import Optional

import networkx as nx

sys.path.append(str(Path(__file__).resolve().parents[3]))
from config import (
    PAPERS_DIR,
    CODE_DIR,
    DATA_DIR
)

# Where we save the graph to disk
GRAPH_PATH = Path(DATA_DIR) / "knowledge_graph.pkl"


# ── Graph Node Types ──────────────────────────────────────────────────────────
# paper       → a published lab paper
# technique   → experimental technique (fiber photometry, DeepLabCut, etc.)
# paradigm    → behavioral paradigm (SAA, LTM, fear conditioning, etc.)
# brain_region→ brain region (VTA, amygdala, etc.)
# concept     → general neuroscience concept
# code_file   → a pipeline Python file
# function    → a function inside a code file
# ─────────────────────────────────────────────────────────────────────────────


def build_papers_graph(G: nx.DiGraph) -> nx.DiGraph:
    """
    Adds lab paper nodes and their known relationships to the graph.
    We define these manually from what we know about the lab's work
    rather than trying to extract them from PDFs (which is unreliable).
    """

    # ── Core lab entities ─────────────────────────────────────────────────────
    G.add_node("Shrestha_Lab",
               type="lab",
               label="Shrestha Lab",
               description="Department of Neurobiology & Behavior, Stony Brook University")

    G.add_node("Prof_Shrestha",
               type="person",
               label="Prof. Prerana Shrestha",
               description="Principal Investigator, Shrestha Lab")

    G.add_edge("Prof_Shrestha", "Shrestha_Lab", relation="leads")

    # ── Techniques ────────────────────────────────────────────────────────────
    techniques = [
        ("fiber_photometry", "Fiber Photometry",
         "Measures calcium/dopamine signals in freely moving animals"),
        ("deeplabcut", "DeepLabCut",
         "Markerless pose estimation for behavioral tracking"),
        ("fear_conditioning", "Fear Conditioning",
         "Associative learning paradigm using aversive stimuli"),
    ]
    for node_id, label, desc in techniques:
        G.add_node(node_id, type="technique", label=label, description=desc)
        G.add_edge("Shrestha_Lab", node_id, relation="uses_technique")

    # ── Behavioral paradigms ──────────────────────────────────────────────────
    paradigms = [
        ("SAA", "Signaled Active Avoidance",
         "Animal learns to avoid aversive stimulus by shuttling"),
        ("LTM", "Long Term Memory",
         "Tests retention of learned associations over time"),
        ("PTC", "Pavlovian Threat Conditioning",
         "Classical conditioning with threat stimuli"),
        ("open_field", "Open Field Test",
         "Measures locomotion and anxiety in novel environment"),
        ("NOR", "Novel Object Recognition",
         "Tests recognition memory using novel objects"),
    ]
    for node_id, label, desc in paradigms:
        G.add_node(node_id, type="paradigm", label=label, description=desc)
        G.add_edge("Shrestha_Lab", node_id, relation="studies_paradigm")

    # ── Brain regions ─────────────────────────────────────────────────────────
    brain_regions = [
        ("VTA", "Ventral Tegmental Area",
         "Dopaminergic brain region involved in reward and aversion"),
        ("amygdala", "Amygdala",
         "Key region for fear learning and emotional memory"),
        ("NAc", "Nucleus Accumbens",
         "Reward circuitry, receives dopamine from VTA"),
        ("prefrontal_cortex", "Prefrontal Cortex",
         "Higher-order cognition and top-down control of fear"),
    ]
    for node_id, label, desc in brain_regions:
        G.add_node(node_id, type="brain_region", label=label, description=desc)
        G.add_edge("Shrestha_Lab", node_id, relation="studies_brain_region")

    # ── Neurotransmitters / signals ───────────────────────────────────────────
    signals = [
        ("dopamine", "Dopamine",
         "Neuromodulator involved in reward, aversion, and learning"),
        ("calcium_signals", "Calcium Signals",
         "Proxy for neural activity measured via fiber photometry"),
        ("dLight", "dLight1.3b",
         "Genetically encoded dopamine sensor used in fiber photometry"),
    ]
    for node_id, label, desc in signals:
        G.add_node(node_id, type="concept", label=label, description=desc)

    G.add_edge("fiber_photometry", "calcium_signals", relation="measures")
    G.add_edge("fiber_photometry", "dopamine", relation="can_measure")
    G.add_edge("dLight", "dopamine", relation="sensor_for")
    G.add_edge("VTA", "dopamine", relation="produces")
    G.add_edge("VTA", "NAc", relation="projects_to")

    # ── Papers ────────────────────────────────────────────────────────────────
    papers = [
        ("Lab_Paper1", "Lab Paper 1", ["fiber_photometry", "SAA", "dopamine"]),
        ("Lab_Paper2", "Lab Paper 2", ["fiber_photometry", "fear_conditioning"]),
        ("Lab_Paper3", "Lab Paper 3", ["SAA", "VTA", "dopamine"]),
        ("Lab_Paper4", "Lab Paper 4", ["LTM", "amygdala"]),
        ("Lab_Paper5", "Lab Paper 5", ["PTC", "fear_conditioning"]),
        ("Lab_Paper6", "Lab Paper 6", ["fiber_photometry", "calcium_signals"]),
        ("Lab_Paper7", "Lab Paper 7", ["deeplabcut", "open_field"]),
        ("Lab_Paper8", "Lab Paper 8", ["NOR", "open_field"]),
        ("Lab_Paper9", "Lab Paper 9", ["SAA", "fiber_photometry", "VTA"]),
        ("Lab_Paper10", "Lab Paper 10", ["dopamine", "NAc"]),
    ]
    for paper_id, label, topics in papers:
        G.add_node(paper_id,
                   type="paper",
                   label=label,
                   source_file=f"{paper_id}.pdf")
        G.add_edge("Shrestha_Lab", paper_id, relation="published")
        for topic in topics:
            if G.has_node(topic):
                G.add_edge(paper_id, topic, relation="studies")

    return G


def build_code_graph(G: nx.DiGraph) -> nx.DiGraph:
    """
    Adds pipeline code nodes and their relationships to the graph.
    Maps out which files do what and how they relate to each other.
    """

    # ── Pipeline entry point ──────────────────────────────────────────────────
    G.add_node("run_analysis.py",
               type="code_file",
               label="run_analysis.py",
               description="Main entry point for the fiber photometry analysis pipeline")

    # ── Core pipeline files ───────────────────────────────────────────────────
    code_files = [
        ("base_mouse.py", "Base Mouse class with shared data loading and preprocessing"),
        ("saa_mouse.py", "SAA-specific mouse analysis — extends BaseMouse"),
        ("snippet_extractor.py", "Extracts peri-event time snippets from photometry signals"),
        ("peth_plotter.py", "Plots peri-event time histograms (PETH)"),
        ("overlay_peth_plotter.py", "Overlays multiple PETH plots for comparison"),
        ("heatmap_plotter.py", "Generates heatmap visualizations of neural activity"),
        ("group_analyzer.py", "Runs analysis across groups of mice"),
        ("metrics_calculator.py", "Calculates behavioral and neural metrics"),
        ("excel_writer.py", "Writes results to Excel output files"),
        ("ymin_excel_writer.py", "Writes Y-min metrics to Excel"),
        ("preprocessing.py", "Signal preprocessing — filtering, normalization"),
        ("tdt_loader.py", "Loads TDT fiber photometry data files"),
        ("behavioral_csv_parser.py", "Parses behavioral CSV data"),
        ("cohort_parser.py", "Parses cohort configuration files"),
        ("folder_scanner.py", "Scans folder structure for experiment data"),
        ("path_utils.py", "Path and file utility functions"),
        ("validators.py", "Input validation functions"),
        ("logging_config.py", "Logging configuration"),
        ("plot_styles.py", "Shared plot styling constants"),
        ("trial_classifier.py", "Classifies trial types (CS+, CS-, AR, etc.)"),
        ("trial_history_analyzer.py", "Analyzes trial history patterns"),
        ("post_ar_shuttle_extractor.py", "Extracts post-avoidance response shuttling data"),
        ("shuttling_spreadsheet_writer.py", "Writes shuttling data to spreadsheet"),
    ]

    for filename, description in code_files:
        if not G.has_node(filename):
            G.add_node(filename,
                       type="code_file",
                       label=filename,
                       description=description)

    # ── File dependency relationships ─────────────────────────────────────────
    dependencies = [
        ("run_analysis.py", "saa_mouse.py", "runs"),
        ("run_analysis.py", "group_analyzer.py", "runs"),
        ("saa_mouse.py", "base_mouse.py", "extends"),
        ("saa_mouse.py", "snippet_extractor.py", "uses"),
        ("saa_mouse.py", "peth_plotter.py", "uses"),
        ("saa_mouse.py", "overlay_peth_plotter.py", "uses"),
        ("saa_mouse.py", "heatmap_plotter.py", "uses"),
        ("saa_mouse.py", "excel_writer.py", "uses"),
        ("saa_mouse.py", "ymin_excel_writer.py", "uses"),
        ("saa_mouse.py", "metrics_calculator.py", "uses"),
        ("saa_mouse.py", "trial_classifier.py", "uses"),
        ("saa_mouse.py", "trial_history_analyzer.py", "uses"),
        ("saa_mouse.py", "post_ar_shuttle_extractor.py", "uses"),
        ("base_mouse.py", "tdt_loader.py", "uses"),
        ("base_mouse.py", "preprocessing.py", "uses"),
        ("base_mouse.py", "behavioral_csv_parser.py", "uses"),
        ("base_mouse.py", "validators.py", "uses"),
        ("base_mouse.py", "path_utils.py", "uses"),
        ("base_mouse.py", "logging_config.py", "uses"),
        ("group_analyzer.py", "cohort_parser.py", "uses"),
        ("group_analyzer.py", "folder_scanner.py", "uses"),
        ("peth_plotter.py", "plot_styles.py", "uses"),
        ("heatmap_plotter.py", "plot_styles.py", "uses"),
        ("excel_writer.py", "shuttling_spreadsheet_writer.py", "uses"),
    ]

    for source, target, relation in dependencies:
        if G.has_node(source) and G.has_node(target):
            G.add_edge(source, target, relation=relation)

    # ── Connect pipeline concepts to neuroscience concepts ────────────────────
    concept_connections = [
        ("snippet_extractor.py", "fiber_photometry", "processes_data_from"),
        ("peth_plotter.py", "fiber_photometry", "visualizes"),
        ("saa_mouse.py", "SAA", "implements_analysis_for"),
        ("base_mouse.py", "fiber_photometry", "loads_data_from"),
        ("tdt_loader.py", "fiber_photometry", "reads_raw_data_from"),
        ("preprocessing.py", "calcium_signals", "processes"),
        ("trial_classifier.py", "SAA", "classifies_trials_for"),
    ]

    for source, target, relation in concept_connections:
        if G.has_node(source) and G.has_node(target):
            G.add_edge(source, target, relation=relation)

    return G


def save_graph(G: nx.DiGraph, path: Path = GRAPH_PATH) -> None:
    """Saves the graph to disk as a pickle file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f)
    print(f"💾 Graph saved to {path}")


def load_graph(path: Path = GRAPH_PATH) -> Optional[nx.DiGraph]:
    """Loads the graph from disk. Returns None if not found."""
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def build_full_graph() -> nx.DiGraph:
    """
    Builds the complete knowledge graph combining
    lab papers and pipeline code.
    """
    print("=" * 55)
    print("  NeuroAssist — Knowledge Graph Builder")
    print("=" * 55)

    G = nx.DiGraph()

    print("\n🧠 Building papers knowledge graph...")
    G = build_papers_graph(G)
    print(f"   ✅ Papers graph built")

    print("\n💻 Building code knowledge graph...")
    G = build_code_graph(G)
    print(f"   ✅ Code graph built")

    # ── Stats ─────────────────────────────────────────────────────────────────
    print(f"\n📊 Graph Statistics:")
    print(f"   Total nodes : {G.number_of_nodes()}")
    print(f"   Total edges : {G.number_of_edges()}")

    # Count by type
    type_counts = {}
    for _, data in G.nodes(data=True):
        node_type = data.get("type", "unknown")
        type_counts[node_type] = type_counts.get(node_type, 0) + 1

    print(f"\n   Node breakdown:")
    for node_type, count in sorted(type_counts.items()):
        print(f"   {'':3}{node_type:<20}: {count}")

    # ── Save ──────────────────────────────────────────────────────────────────
    print()
    save_graph(G)

    # ── Quick query test ──────────────────────────────────────────────────────
    print("\n🔍 Test queries:")

    # What techniques does the lab use?
    lab_techniques = [
        G.nodes[n]["label"]
        for _, n, d in G.out_edges("Shrestha_Lab", data=True)
        if d.get("relation") == "uses_technique"
    ]
    print(f"   Lab techniques  : {lab_techniques}")

    # What paradigms does the lab study?
    lab_paradigms = [
        G.nodes[n]["label"]
        for _, n, d in G.out_edges("Shrestha_Lab", data=True)
        if d.get("relation") == "studies_paradigm"
    ]
    print(f"   Lab paradigms   : {lab_paradigms}")

    # What files does saa_mouse.py use?
    saa_deps = [
        n for _, n, d in G.out_edges("saa_mouse.py", data=True)
        if d.get("relation") == "uses"
    ]
    print(f"   saa_mouse.py uses: {saa_deps}")

    print("\n✅ Knowledge graph is ready!")
    return G


if __name__ == "__main__":
    build_full_graph()