"""
Methode 3: Echo Graph Analysis (EGA)
====================================

Konzept: Fuer jeden Zustand finde alle zukuenftigen Schritte die "nahe"
zurueckkehren. Baue einen Graphen der Zustaende mit ihren "Echos" verbindet.

Frage beantwortet:
- #3 (Zustands-Echo, strukturelle Erinnerung)

Hypothese: Training besucht aehnliche Zustaende mehrfach ohne exakte
Wiederholung - eine Form von "struktureller Erinnerung".

Autor: Maximilian (MI-Forschung)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
import networkx as nx
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import os

from .common import EmbeddingReader, EMBEDDINGS_FILE, load_vocabulary
from .common.utils import ensure_output_dir


def sample_embeddings(emb_reader, sample_interval=100, max_snapshots=None,
                      word_idx=None):
    """
    Samplet Embedding-Snapshots in regelmaessigen Abstaenden.

    Args:
        emb_reader: EmbeddingReader Instanz
        sample_interval: Jeder n-te Snapshot
        max_snapshots: Maximale Anzahl (None = alle)
        word_idx: Bestimmtes Wort (None = alle Woerter flatten)

    Returns:
        sampled: numpy Array der Embeddings
        indices: Originale Snapshot-Indizes
    """
    total = min(emb_reader.num_snapshots, max_snapshots) if max_snapshots else emb_reader.num_snapshots
    indices = list(range(0, total, sample_interval))

    print(f"  Sampling {len(indices)} Snapshots (Interval={sample_interval})...")

    if word_idx is not None:
        # Nur ein Wort
        sampled = emb_reader.read_word_trajectory(word_idx, 0, total, sample_interval)
    else:
        # Alle Woerter (flatten)
        sampled = []
        for i, idx in enumerate(indices):
            snapshot = emb_reader.read_snapshot(idx)
            sampled.append(snapshot.flatten())  # (vocab_size * embedding_dim,)
            if (i + 1) % 1000 == 0:
                print(f"    {i+1:,} / {len(indices):,}")
        sampled = np.array(sampled)

    return sampled, np.array(indices)


def build_echo_graph(sampled, indices, threshold=0.5, min_gap=1000):
    """
    Baut den Echo-Graphen mit BallTree fuer effiziente Nachbarsuche.

    Args:
        sampled: Gesamplete Embeddings
        indices: Originale Snapshot-Indizes
        threshold: Maximale Distanz fuer Echo
        min_gap: Minimaler zeitlicher Abstand

    Returns:
        edges: Liste von (idx_i, idx_j, distance) Tupeln
        graph: NetworkX Graph
    """
    print(f"  Baue Echo-Graph (threshold={threshold}, min_gap={min_gap})...")

    # BallTree fuer effiziente Nachbarsuche
    tree = BallTree(sampled)

    edges = []
    for i in range(len(sampled)):
        # Finde alle Nachbarn innerhalb threshold
        neighbors = tree.query_radius([sampled[i]], r=threshold)[0]

        for j in neighbors:
            # Nur zukuenftige Echos mit min_gap
            if indices[j] > indices[i] + min_gap:
                dist = np.linalg.norm(sampled[i] - sampled[j])
                edges.append((int(indices[i]), int(indices[j]), float(dist)))

        if (i + 1) % 1000 == 0:
            print(f"    {i+1:,} / {len(sampled):,}")

    # NetworkX Graph erstellen
    graph = nx.DiGraph()
    for idx in indices:
        graph.add_node(int(idx))
    for i, j, d in edges:
        graph.add_edge(i, j, weight=d)

    print(f"  Gefundene Echos: {len(edges)}")
    return edges, graph


def analyze_threshold_sensitivity(sampled, indices, min_gap=1000, thresholds=None):
    """
    Testet verschiedene Thresholds und gibt Echo-Counts zurueck.

    Args:
        sampled: Gesamplete Embeddings
        indices: Snapshot-Indizes
        min_gap: Minimaler zeitlicher Abstand
        thresholds: Liste von Thresholds zum Testen

    Returns:
        sensitivity: Dict {threshold: n_edges}
    """
    if thresholds is None:
        thresholds = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

    print(f"  Threshold-Sensitivity Analyse...")
    sensitivity = {}
    tree = BallTree(sampled)

    # Effizient: Query mit max threshold, dann filtern
    max_thresh = max(thresholds)
    all_neighbors = tree.query_radius(sampled, r=max_thresh)

    for thresh in sorted(thresholds):
        n_edges = 0
        for i, neighbors in enumerate(all_neighbors):
            for j in neighbors:
                if indices[j] > indices[i] + min_gap:
                    dist = np.linalg.norm(sampled[i] - sampled[j])
                    if dist <= thresh:
                        n_edges += 1
        sensitivity[thresh] = n_edges
        print(f"    Threshold {thresh}: {n_edges:,} Echos")

    return sensitivity


# Globale Variable fuer Worker-Prozesse (Multiprocessing)
_worker_successors = None


def _init_chain_worker(graph_edges):
    """Initialisiert successors-Dict einmal pro Worker."""
    global _worker_successors
    _worker_successors = defaultdict(list)
    for src, dst in graph_edges:
        _worker_successors[src].append(dst)


def _find_chains_from_node(args):
    """Worker-Funktion fuer parallele Chain-Suche."""
    start_node, max_chain_length, min_chain_length = args
    global _worker_successors

    chains = []
    stack = [(start_node, [start_node])]
    paths_found = 0

    while stack and paths_found < 100:
        node, path = stack.pop()
        for successor in _worker_successors[node]:
            if successor in path:
                continue
            if len(path) >= max_chain_length:
                continue
            new_path = path + [successor]
            if len(new_path) >= min_chain_length:
                chains.append(new_path)
                paths_found += 1
                if paths_found >= 100:
                    break
            stack.append((successor, new_path))

    return chains


def _find_chains_sequential(graph, start_nodes, min_chain_length,
                            max_chain_length, max_chains):
    """Urspruengliche sequentielle Implementierung als Fallback."""
    chains = []

    for start_node in start_nodes:
        if len(chains) >= max_chains:
            break

        stack = [(start_node, [start_node])]
        paths_from_node = 0

        while stack and paths_from_node < 100:
            node, path = stack.pop()

            for successor in graph.successors(node):
                if successor in path:
                    continue
                if len(path) >= max_chain_length:
                    continue

                new_path = path + [successor]
                if len(new_path) >= min_chain_length:
                    chains.append(new_path)
                    paths_from_node += 1
                    if paths_from_node >= 100:
                        break
                stack.append((successor, new_path))

    chains.sort(key=len, reverse=True)
    return chains[:max_chains]


def find_echo_chains(graph, min_chain_length=3, max_chain_length=10,
                     max_chains=1000, n_workers=None):
    """
    Findet Echo-Ketten: Sequenzen i -> j -> k -> ...
    wo der Zustand immer wieder zurueckkehrt.

    Bei kleinen Graphen: Sequentiell (weniger Overhead)
    Bei grossen Graphen: Parallel mit multiprocessing
    """
    nodes = list(graph.nodes())

    # Sortiere Startknoten nach Out-Degree (vielversprechendste zuerst)
    nodes.sort(key=lambda n: graph.out_degree(n), reverse=True)

    # Begrenze Startknoten (Top 500 oder alle bei kleinen Graphen)
    max_starts = min(len(nodes), 500)

    # Bei kleinen Graphen: Sequentiell (schneller wegen Overhead)
    if len(graph.edges()) < 1000:
        print(f"    Sequentielle Chain-Suche ({len(graph.edges())} Kanten)...")
        return _find_chains_sequential(graph, nodes[:max_starts],
                                        min_chain_length, max_chain_length, max_chains)

    # Parallel
    graph_edges = list(graph.edges())
    args = [(node, max_chain_length, min_chain_length)
            for node in nodes[:max_starts]]

    n_workers = n_workers or max(1, cpu_count() - 1)
    print(f"    Parallele Chain-Suche mit {n_workers} Workers ({len(graph.edges())} Kanten)...")

    with Pool(n_workers, initializer=_init_chain_worker,
              initargs=(graph_edges,)) as pool:
        results = pool.map(_find_chains_from_node, args)

    # Flatten und sortieren
    all_chains = [chain for chains in results for chain in chains]
    all_chains.sort(key=len, reverse=True)

    return all_chains[:max_chains]


def compute_echo_depth(edges):
    """
    Berechnet Echo-Tiefe: Wie weit in der Vergangenheit reichen Echos?

    Returns:
        depths: Dict mapping step_idx -> max echo depth
        max_depth: Maximale Echo-Tiefe
        depth_distribution: Histogramm der Tiefen
    """
    depths = defaultdict(int)

    for i, j, _ in edges:
        depth = j - i  # Zeitlicher Abstand
        depths[i] = max(depths[i], depth)

    if depths:
        max_depth = max(depths.values())
        depth_values = list(depths.values())
    else:
        max_depth = 0
        depth_values = []

    return dict(depths), max_depth, depth_values


def analyze_echo_clustering(edges, indices, steps_per_epoch):
    """
    Analysiert ob Echos in bestimmten Epochen clustern.

    Args:
        edges: Liste von (source, target, distance) Tupeln
        indices: Snapshot-Indizes
        steps_per_epoch: Anzahl Schritte pro Epoch (aus config)

    Returns:
        echo_by_epoch: Dict {epoch: [(i, j, d), ...]}
        epoch_counts: Dict {epoch: count}
    """
    echo_by_epoch = defaultdict(list)

    for i, j, d in edges:
        source_epoch = i // steps_per_epoch
        echo_by_epoch[source_epoch].append((i, j, d))

    # Zaehle Echos pro Epoch
    epoch_counts = {epoch: len(echos) for epoch, echos in echo_by_epoch.items()}

    return echo_by_epoch, epoch_counts


def compute_echo_divergence(sampled, edges, indices):
    """
    Analysiert wie sich Echo-verbundene Zustaende unterscheiden.

    Bei aehnlicher Distanz: Welche Dimensionen divergieren am meisten?
    """
    if len(edges) == 0:
        return None

    divergences = []
    idx_to_sample = {int(idx): i for i, idx in enumerate(indices)}

    for i, j, d in edges[:1000]:  # Sample der Echos
        if i in idx_to_sample and j in idx_to_sample:
            sample_i = sampled[idx_to_sample[i]]
            sample_j = sampled[idx_to_sample[j]]
            diff = np.abs(sample_i - sample_j)
            divergences.append(diff)

    if divergences:
        divergences = np.array(divergences)
        mean_divergence = np.mean(divergences, axis=0)
        return mean_divergence
    return None


def visualize_echo_graph(graph, edges, indices, sensitivity, output_path, max_nodes=500):
    """Visualisiert den Echo-Graphen."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Echo-Netzwerk (Subsample)
    ax = axes[0, 0]
    if len(graph.nodes()) > max_nodes:
        # Subsample: Nur Knoten mit Echos
        nodes_with_edges = set()
        for i, j, _ in edges[:max_nodes]:
            nodes_with_edges.add(i)
            nodes_with_edges.add(j)
        subgraph = graph.subgraph(list(nodes_with_edges)[:max_nodes])
    else:
        subgraph = graph

    if len(subgraph.nodes()) > 0:
        pos = {node: (node % 100, node // 100) for node in subgraph.nodes()}
        nx.draw_networkx_nodes(subgraph, pos, ax=ax, node_size=10, alpha=0.5)
        nx.draw_networkx_edges(subgraph, pos, ax=ax, alpha=0.2, arrows=True,
                               arrowsize=5, edge_color='blue')
    ax.set_title(f'Echo-Netzwerk ({len(subgraph.nodes())} Knoten)')
    ax.axis('off')

    # 2. Echo-Tiefe Distribution
    ax = axes[0, 1]
    _, max_depth, depth_values = compute_echo_depth(edges)
    if depth_values:
        ax.hist(depth_values, bins=50, color='purple', alpha=0.7)
        ax.axvline(np.median(depth_values), color='red', linestyle='--',
                   label=f'Median: {np.median(depth_values):.0f}')
        ax.set_xlabel('Echo-Tiefe (Schritte)')
        ax.set_ylabel('Haeufigkeit')
        ax.set_title(f'Echo-Tiefe Verteilung (Max: {max_depth:,})')
        ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Threshold Sensitivity
    ax = axes[0, 2]
    if sensitivity:
        thresholds = sorted(sensitivity.keys())
        counts = [sensitivity[t] for t in thresholds]
        ax.plot(thresholds, counts, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Anzahl Echos')
        ax.set_title('Threshold-Sensitivity')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    # 4. Echos ueber Zeit
    ax = axes[1, 0]
    if edges:
        source_times = [e[0] for e in edges]
        target_times = [e[1] for e in edges]
        ax.scatter(source_times, target_times, alpha=0.1, s=1)
        ax.plot([0, max(target_times)], [0, max(target_times)], 'r--', alpha=0.5)
        ax.set_xlabel('Quell-Schritt')
        ax.set_ylabel('Ziel-Schritt (Echo)')
        ax.set_title('Echo-Verbindungen ueber Zeit')
    ax.grid(True, alpha=0.3)

    # 5. Echo-Dichte pro Epoch (placeholder - wird spaeter gefuellt)
    ax = axes[1, 1]
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Anzahl Echos')
    ax.set_title('Echo-Dichte pro Epoch')
    ax.grid(True, alpha=0.3)

    # 6. Zusammenfassung
    ax = axes[1, 2]
    ax.axis('off')
    summary = f"""
    ECHO GRAPH ANALYSIS (EGA)
    =========================

    GRAPH:
    - Knoten: {len(graph.nodes()):,}
    - Kanten (Echos): {len(edges):,}
    - Dichte: {nx.density(graph):.6f}

    THRESHOLD-SENSITIVITY:
    {chr(10).join(f'  {t}: {c:,} Echos' for t, c in sorted(sensitivity.items()))}

    ECHO-TIEFE:
    - Maximum: {max_depth:,} Schritte
    - Median: {np.median(depth_values):,.0f} if depth_values else 0
    """
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  Gespeichert: {output_path}")
    plt.close()


def visualize_echo_chains(chains, output_path):
    """Visualisiert die laengsten Echo-Ketten."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Top 20 Ketten
    for i, chain in enumerate(chains[:20]):
        y = [i] * len(chain)
        ax.plot(chain, y, 'o-', markersize=4, alpha=0.7,
                label=f'Kette {i+1} (Laenge {len(chain)})')

    ax.set_xlabel('Trainingsschritt')
    ax.set_ylabel('Ketten-Rang')
    ax.set_title(f'Top 20 Echo-Ketten')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  Gespeichert: {output_path}")
    plt.close()


def run_analysis(sample_interval=100, threshold=1.0, min_gap=1000, max_snapshots=None):
    """
    Fuehrt die vollstaendige EGA-Analyse durch.

    Args:
        sample_interval: Sampling-Intervall fuer Snapshots
        threshold: Distanz-Schwelle fuer Echos
        min_gap: Minimaler zeitlicher Abstand fuer Echos
        max_snapshots: Maximale Snapshot-Anzahl

    Returns:
        dict mit allen Ergebnissen
    """
    output_dir = ensure_output_dir()
    print("=" * 70)
    print("ECHO GRAPH ANALYSIS (EGA)")
    print("=" * 70)

    # Vokabular laden
    vocab_data = load_vocabulary()
    config = vocab_data['config']
    steps_per_epoch = config['num_pairs']  # 420

    print(f"\nKonfiguration:")
    print(f"  Sample-Intervall: {sample_interval}")
    print(f"  Echo-Threshold: {threshold}")
    print(f"  Min Gap: {min_gap}")
    print(f"  Steps per Epoch: {steps_per_epoch}")

    with EmbeddingReader(EMBEDDINGS_FILE) as reader:
        # 1. Embeddings samplen
        sampled, indices = sample_embeddings(
            reader, sample_interval, max_snapshots
        )
        print(f"\n  Gesampelte Embeddings: {len(sampled)}")
        print(f"  Dimensionen pro Sample: {sampled.shape[1]}")

        # 2. Threshold-Sensitivity Analyse
        print("\n  Analysiere Threshold-Sensitivity...")
        sensitivity = analyze_threshold_sensitivity(sampled, indices, min_gap)

        # 3. Echo-Graph bauen (mit gewaehltem threshold)
        edges, graph = build_echo_graph(sampled, indices, threshold, min_gap)

        # 4. Echo-Ketten finden
        print("\n  Suche Echo-Ketten...")
        chains = find_echo_chains(graph)
        print(f"  Gefundene Ketten: {len(chains)}")
        if chains:
            print(f"  Laengste Kette: {len(chains[0])} Schritte")

        # 5. Echo-Tiefe analysieren
        depths, max_depth, depth_values = compute_echo_depth(edges)
        print(f"\n  Maximale Echo-Tiefe: {max_depth:,} Schritte")
        if depth_values:
            print(f"  Mediane Echo-Tiefe: {np.median(depth_values):,.0f} Schritte")

        # 6. Echo-Clustering (mit korrektem steps_per_epoch)
        echo_by_epoch, epoch_counts = analyze_echo_clustering(edges, indices, steps_per_epoch)
        if epoch_counts:
            top_epoch = max(epoch_counts.keys(), key=epoch_counts.get)
            print(f"\n  Epoch mit meisten Echos: {top_epoch} ({epoch_counts[top_epoch]} Echos)")

        # 7. Echo-Divergenz
        print("\n  Analysiere Echo-Divergenz...")
        divergence = compute_echo_divergence(sampled, edges, indices)

        # 8. Visualisierungen
        print("\n  Erstelle Visualisierungen...")
        visualize_echo_graph(graph, edges, indices, sensitivity,
                             os.path.join(output_dir, 'ega_overview.png'))

        if chains:
            visualize_echo_chains(chains,
                                  os.path.join(output_dir, 'ega_chains.png'))

    # Ergebnisse zusammenfassen
    results = {
        'num_samples': len(sampled),
        'num_edges': len(edges),
        'num_chains': len(chains),
        'max_chain_length': len(chains[0]) if chains else 0,
        'max_echo_depth': max_depth,
        'median_echo_depth': float(np.median(depth_values)) if depth_values else 0,
        'echo_by_epoch': {int(k): len(v) for k, v in echo_by_epoch.items()},
        'graph_density': nx.density(graph) if len(graph.nodes()) > 0 else 0,
        'threshold_sensitivity': {float(k): v for k, v in sensitivity.items()}
    }

    print("\n" + "=" * 70)
    print("EGA ANALYSE ABGESCHLOSSEN")
    print("=" * 70)

    return results


if __name__ == "__main__":
    # Mit Standardparametern ausfuehren
    results = run_analysis(sample_interval=100, threshold=1.0, min_gap=1000)
