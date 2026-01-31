"""
Methode 6: Dimensional Autonomy Index (DAI)
==========================================

Konzept: Miss wie unabhaengig jede Dimension sich aendert. Zwei Dimensionen
sind "gekoppelt" wenn ihre Aenderungen im Vorzeichen korrelieren.
Finde "autonome" Dimensionen und "Leader-Dimensionen".

Frage beantwortet:
- #6 (Verbindungen ohne direkten Vergleich)
- #3 (nicht-kausale Kopplung)

Hypothese: Manche Dimensionen sind "Kontrollkanaele" - implizite Hierarchie
in Parameterupdates.

Autor: Maximilian (MI-Forschung)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from .common import StepReader, STEPS_FILE, load_vocabulary
from .common.utils import ensure_output_dir


def compute_sign_cooccurrence(changes):
    """
    Berechnet Co-occurrence Matrix der Vorzeichen.

    C[i,j] = P(sign(change_i) == sign(change_j))

    Args:
        changes: Array (N, D) der Aenderungsvektoren

    Returns:
        C: Co-occurrence Matrix (D, D)
    """
    signs = np.sign(changes)
    n_dims = signs.shape[1]

    C = np.zeros((n_dims, n_dims))
    for i in range(n_dims):
        for j in range(n_dims):
            C[i, j] = np.mean(signs[:, i] == signs[:, j])

    return C


def compute_autonomy_index(C):
    """
    Berechnet Autonomie-Index fuer jede Dimension.

    Autonomie = 1 - max(Kopplung mit anderen Dimensionen)

    Hohe Autonomie = Dimension aendert sich unabhaengig von allen anderen.

    Args:
        C: Co-occurrence Matrix

    Returns:
        autonomy: Array (D,) mit Autonomie-Werten
    """
    # Entferne Diagonale (Selbst-Korrelation ist immer 1)
    C_masked = C.copy()
    np.fill_diagonal(C_masked, 0)

    # Autonomie = 1 - maximale Kopplung mit einer anderen Dimension
    max_coupling = np.max(C_masked, axis=1)
    autonomy = 1 - max_coupling

    return autonomy


def find_leader_dimensions(changes, max_lag=10):
    """
    Findet "Leader"-Dimensionen: Dimensionen die sich zuerst aendern,
    andere folgen.

    Analysiert zeitversetzte Korrelationen.

    Args:
        changes: Array (N, D)
        max_lag: Maximaler Zeitversatz zu pruefen

    Returns:
        leader_scores: Dict mapping dim -> leader_score
        follow_matrix: Matrix welche Dimension welcher folgt
    """
    signs = np.sign(changes)
    n_dims = changes.shape[1]

    # Berechne Cross-Korrelation mit Zeitversatz
    leader_scores = np.zeros(n_dims)
    follow_matrix = np.zeros((n_dims, n_dims))

    for dim_a in range(n_dims):
        for dim_b in range(n_dims):
            if dim_a == dim_b:
                continue

            # Pruefe ob dim_a dim_b vorauslaeuft
            for lag in range(1, max_lag + 1):
                if lag < len(signs):
                    # Korrelation: sign_a[t] mit sign_b[t+lag]
                    corr = np.mean(signs[:-lag, dim_a] == signs[lag:, dim_b])

                    if corr > 0.6:  # Signifikante Vorauslauf-Korrelation
                        leader_scores[dim_a] += (corr - 0.5) / lag
                        follow_matrix[dim_a, dim_b] += corr

    return leader_scores, follow_matrix


def analyze_autonomy_evolution(step_reader, window_size=10000, step=5000):
    """
    Analysiert wie sich Autonomie ueber das Training entwickelt.

    Returns:
        autonomy_evolution: Array (n_windows, n_dims)
        window_centers: Zeitpunkte der Fenster-Mitten
    """
    n = step_reader.num_steps
    n_dims = step_reader.embedding_dim

    window_centers = []
    autonomy_evolution = []

    for start in range(0, n - window_size, step):
        end = start + window_size

        # Sammle Changes in diesem Fenster
        changes = []
        for i in range(start, end):
            step_data = step_reader.read_step(i)
            changes.append(step_data['change'])

        changes = np.array(changes)

        # Berechne Autonomie
        C = compute_sign_cooccurrence(changes)
        autonomy = compute_autonomy_index(C)

        window_centers.append((start + end) // 2)
        autonomy_evolution.append(autonomy)

    return np.array(autonomy_evolution), np.array(window_centers)


def compute_dimensional_coupling_graph(C, threshold=0.7):
    """
    Erstellt Kopplungs-Graph aus Co-occurrence Matrix.

    Args:
        C: Co-occurrence Matrix
        threshold: Schwelle fuer "gekoppelt"

    Returns:
        edges: Liste von (dim_i, dim_j, coupling) Tupeln
        clusters: Liste von Dimensions-Clustern
    """
    n_dims = len(C)
    edges = []

    for i in range(n_dims):
        for j in range(i + 1, n_dims):
            if C[i, j] > threshold:
                edges.append((i, j, C[i, j]))

    # Einfaches Clustering ueber Union-Find
    parent = list(range(n_dims))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i, j, _ in edges:
        union(i, j)

    # Cluster extrahieren
    clusters = {}
    for i in range(n_dims):
        root = find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)

    return edges, list(clusters.values())


def visualize_results(C, autonomy, leader_scores, follow_matrix, evolution,
                      window_centers, edges, clusters, output_dir):
    """Erstellt Visualisierungen der DAI-Analyse."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Co-occurrence Matrix
    ax = axes[0, 0]
    im = ax.imshow(C, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Dimension')
    ax.set_title('Vorzeichen Co-occurrence Matrix')
    plt.colorbar(im, ax=ax)

    # 2. Autonomie pro Dimension
    ax = axes[0, 1]
    dims = range(len(autonomy))
    colors = plt.cm.RdYlGn(autonomy)
    ax.bar(dims, autonomy, color=colors)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Autonomie-Index')
    ax.set_title('Dimensionale Autonomie')
    ax.axhline(np.mean(autonomy), color='red', linestyle='--',
               label=f'Mittel: {np.mean(autonomy):.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Leader Scores
    ax = axes[0, 2]
    ax.bar(dims, leader_scores, color='purple', alpha=0.7)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Leader Score')
    ax.set_title('Leader-Dimensionen')
    ax.grid(True, alpha=0.3)

    # 4. Follow Matrix
    ax = axes[1, 0]
    im = ax.imshow(follow_matrix, cmap='Blues')
    ax.set_xlabel('Follower Dimension')
    ax.set_ylabel('Leader Dimension')
    ax.set_title('Follow-Beziehungen')
    plt.colorbar(im, ax=ax)

    # 5. Autonomie-Evolution
    ax = axes[1, 1]
    for dim in range(evolution.shape[1]):
        ax.plot(window_centers, evolution[:, dim], alpha=0.7, label=f'Dim {dim}')
    ax.set_xlabel('Trainingsschritt')
    ax.set_ylabel('Autonomie')
    ax.set_title('Autonomie-Evolution ueber Training')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # 6. Zusammenfassung
    ax = axes[1, 2]
    ax.axis('off')

    summary = f"""
    DIMENSIONAL AUTONOMY INDEX - ZUSAMMENFASSUNG

    AUTONOMIE:
    - Mittlere Autonomie: {np.mean(autonomy):.3f}
    - Max Autonomie: {np.max(autonomy):.3f} (Dim {np.argmax(autonomy)})
    - Min Autonomie: {np.min(autonomy):.3f} (Dim {np.argmin(autonomy)})

    LEADER-DIMENSIONEN:
    - Top Leader: Dim {np.argmax(leader_scores)} (Score: {np.max(leader_scores):.3f})
    - Bottom: Dim {np.argmin(leader_scores)} (Score: {np.min(leader_scores):.3f})

    KOPPLUNGS-CLUSTER:
    - Anzahl Cluster: {len(clusters)}
    - Groesster Cluster: {max(len(c) for c in clusters)} Dimensionen
    - Anzahl starke Kopplungen: {len(edges)}

    INTERPRETATION:
    - Hohe Autonomie = unabhaengige Dimension
    - Hoher Leader Score = treibt andere Dimensionen
    - Grosse Cluster = stark verkoppelte Gruppen
    """
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'dai_analysis.png')
    plt.savefig(output_path, dpi=150)
    print(f"  Gespeichert: {output_path}")
    plt.close()


def run_analysis(max_steps=None):
    """
    Fuehrt die vollstaendige DAI-Analyse durch.
    """
    output_dir = ensure_output_dir()
    print("=" * 70)
    print("DIMENSIONAL AUTONOMY INDEX (DAI) ANALYSE")
    print("=" * 70)

    vocab_data = load_vocabulary()
    config = vocab_data['config']

    with StepReader(STEPS_FILE) as reader:
        # 1. Changes laden
        print("\n  Lade Change-Vektoren...")
        n = min(reader.num_steps, max_steps) if max_steps else reader.num_steps
        changes = reader.get_all_changes(n)
        print(f"  Geladen: {len(changes):,} Vektoren")

        # 2. Co-occurrence Matrix
        print("\n  Berechne Co-occurrence Matrix...")
        C = compute_sign_cooccurrence(changes)

        # 3. Autonomie-Index
        autonomy = compute_autonomy_index(C)
        print(f"  Autonomie pro Dimension: {autonomy}")
        print(f"  Mittlere Autonomie: {np.mean(autonomy):.3f}")

        # 4. Leader-Dimensionen
        print("\n  Suche Leader-Dimensionen...")
        leader_scores, follow_matrix = find_leader_dimensions(changes)
        top_leader = np.argmax(leader_scores)
        print(f"  Top Leader: Dimension {top_leader} (Score: {leader_scores[top_leader]:.3f})")

        # 5. Autonomie-Evolution
        print("\n  Analysiere Autonomie-Evolution...")
        evolution, window_centers = analyze_autonomy_evolution(reader)

        # 6. Kopplungs-Graph
        edges, clusters = compute_dimensional_coupling_graph(C)
        print(f"\n  Gefundene Kopplungs-Cluster: {len(clusters)}")
        for i, cluster in enumerate(clusters):
            print(f"    Cluster {i+1}: Dims {cluster}")

        # 7. Visualisierungen
        print("\n  Erstelle Visualisierungen...")
        visualize_results(C, autonomy, leader_scores, follow_matrix, evolution,
                          window_centers, edges, clusters, output_dir)

    # Ergebnisse
    results = {
        'cooccurrence_matrix': C.tolist(),
        'autonomy': autonomy.tolist(),
        'leader_scores': leader_scores.tolist(),
        'top_leader': int(top_leader),
        'clusters': clusters,
        'mean_autonomy': float(np.mean(autonomy))
    }

    print("\n" + "=" * 70)
    print("DAI ANALYSE ABGESCHLOSSEN")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_analysis()
