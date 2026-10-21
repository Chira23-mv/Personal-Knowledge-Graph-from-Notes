"""
Personal Knowledge Graph from Notes
-----------------------------------
Reads a folder of .txt / .md notes, finds related notes with TF-IDF similarity,
and draws an interactive graph (like Obsidian's graph view).

  CLI  :  python 4_notes_knowledge_graph.py path/to/notes
          python 4_notes_knowledge_graph.py --demo        (tries it on sample notes)
  Web  :  streamlit run 4_notes_knowledge_graph.py

Install:  pip install scikit-learn networkx pyvis streamlit
"""
import sys
import tempfile
from pathlib import Path

import networkx as nx
from pyvis.network import Network
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PALETTE = ["#4cc9f0", "#f72585", "#b5179e", "#7209b7", "#3a86ff", "#06d6a0", "#ffd166", "#ef476f"]

SAMPLE = {
    "python_basics.md": "Python lists dictionaries functions loops. Python is great for scripting and data analysis.",
    "pandas_notes.md": "Pandas dataframes for data analysis in Python. Cleaning data, groupby, merge and plots.",
    "machine_learning.md": "Machine learning models train on data. Regression, classification, overfitting, scikit-learn.",
    "neural_nets.md": "Neural networks are machine learning models with layers, weights and backpropagation training.",
    "docker_intro.md": "Docker containers package applications. Images, volumes, Dockerfile and deployment.",
    "ci_cd.md": "CI/CD pipelines automate testing and deployment of containers and applications with GitHub Actions.",
    "recipe_biryani.md": "Biryani recipe: basmati rice, spices, marinated chicken, saffron, slow cooking.",
}


def load_notes(folder: Path) -> dict:
    notes = {}
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in {".txt", ".md"} and p.is_file():
            text = p.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                notes[p.stem] = text
    return notes


def build_graph_html(notes: dict, threshold: float = 0.15, top_k: int = 5) -> tuple:
    names = list(notes)
    if len(names) < 2:
        raise ValueError("Need at least 2 non-empty notes.")

    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    X = vec.fit_transform(notes.values())
    terms = vec.get_feature_names_out()
    sim = cosine_similarity(X)

    keywords = {}
    for i, name in enumerate(names):
        row = X[i].toarray()[0]
        keywords[name] = [terms[j] for j in row.argsort()[::-1][:top_k] if row[j] > 0]

    G = nx.Graph()
    G.add_nodes_from(names)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if sim[i, j] >= threshold:
                G.add_edge(names[i], names[j], weight=float(sim[i, j]))
        # make sure nobody floats alone if they have even a weak neighbour
        if G.degree(names[i]) == 0:
            others = [(sim[i, j], j) for j in range(len(names)) if j != i]
            best, j = max(others)
            if best > 0.05:
                G.add_edge(names[i], names[j], weight=float(best))

    colour_of = {}
    if G.number_of_edges():
        for c, group in enumerate(nx.algorithms.community.greedy_modularity_communities(G, weight="weight")):
            for n in group:
                colour_of[n] = PALETTE[c % len(PALETTE)]

    net = Network(height="700px", width="100%", bgcolor="#0e1117", font_color="white")
    for n in names:
        net.add_node(
            n, label=n, color=colour_of.get(n, "#888888"),
            size=15 + 6 * G.degree(n),
            title=f"<b>{n}</b><br>Keywords: {', '.join(keywords[n])}<br><br>{notes[n][:200]}...",
        )
    for a, b, d in G.edges(data=True):
        shared = sorted(set(keywords[a]) & set(keywords[b]))
        net.add_edge(a, b, value=d["weight"] * 10,
                     title=f"similarity {d['weight']:.2f}" + (f" | shared: {', '.join(shared)}" if shared else ""))
    net.barnes_hut(gravity=-8000, spring_length=140)

    out = Path(tempfile.gettempdir()) / "knowledge_graph.html"
    net.save_graph(str(out))
    html = out.read_text(encoding="utf-8", errors="ignore")
    return html, out, G, keywords


def run_cli(arg: str) -> None:
    if arg == "--demo":
        notes = SAMPLE
    else:
        notes = load_notes(Path(arg))
    html, out, G, keywords = build_graph_html(notes)
    final = Path("knowledge_graph.html")
    final.write_text(html, encoding="utf-8")
    print(f"✅ {G.number_of_nodes()} notes, {G.number_of_edges()} links -> open {final.resolve()}")
    for n, kws in keywords.items():
        print(f"  {n}: {', '.join(kws)}")


def run_web() -> None:
    import streamlit as st
    import streamlit.components.v1 as components

    st.set_page_config(page_title="Notes Knowledge Graph", layout="wide", page_icon="🕸️")
    st.title("🕸️ Personal Knowledge Graph")
    folder = st.sidebar.text_input("Notes folder", "")
    threshold = st.sidebar.slider("Link threshold (higher = fewer links)", 0.05, 0.6, 0.15, 0.01)
    demo = st.sidebar.checkbox("Use sample notes", value=not folder)

    notes = SAMPLE if demo else (load_notes(Path(folder)) if folder and Path(folder).is_dir() else {})
    if not notes:
        st.info("Enter a folder with .txt/.md notes, or tick 'Use sample notes'.")
        return
    try:
        html, _, G, keywords = build_graph_html(notes, threshold)
    except ValueError as e:
        st.error(str(e))
        return
    st.caption(f"{G.number_of_nodes()} notes · {G.number_of_edges()} links · hover nodes for keywords, drag to explore")
    components.html(html, height=720, scrolling=False)
    with st.expander("Keywords per note"):
        for n, kws in keywords.items():
            st.write(f"**{n}**: {', '.join(kws)}")


def _in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _in_streamlit():
    run_web()
elif __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 4_notes_knowledge_graph.py <notes_folder>  |  --demo")
    else:
        run_cli(sys.argv[1])
