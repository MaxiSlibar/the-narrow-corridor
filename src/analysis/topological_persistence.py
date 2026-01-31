"""
Methode 16: Topological Persistence Analysis (TPA)
=================================================

Konzept: Beziehungen durch Topologie definieren, nicht Koordinaten.
Zwei Woerter sind "topologisch verbunden" wenn ihre Trajektorien
aehnliche Komplexitaet/Struktur haben.

Ansatz (ohne gudhi - Fallback):
- Anzahl Richtungswechsel (Kruemmungs-Nulldurchgaenge)
- Selbst-Kreuzungs-Index (wann kommt Trajektorie sich selbst nahe?)
- Fraktale Dimension Approximation

Frage beantwortet:
- Gibt es "topologisch aehnliche" Woerter?
- Korreliert topologische Komplexitaet mit Wortklasse?

Autor: Maximilian (MI-Forschung)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict
import os

from .common import EmbeddingReader, EMBEDDINGS_FILE, load_vocabulary
from .common.utils import ensure_output_dir

# Versuche gudhi zu importieren (optional)
try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False
    print("  [TPA] gudhi nicht verfuegbar - nutze Fallback-Metriken")


def count_direction_changes(trajectory):
    """
    Zaehlt wie oft die Trajektorie ihre Richtung aendert.

    Returns:
        n_changes: Anzahl Richtungswechsel
        change_rate: Aenderungsrate (pro Schritt)
    """
    if len(trajectory) < 3:
        return 0, 0.0

    # Geschwindigkeiten (Differenzen)
    velocities = np.diff(trajectory, axis=0)

    # Skalarprodukt aufeinanderfolgender Geschwindigkeiten
    dot_products = np.sum(velocities[:-1] * velocities[1:], axis=1)

    # Richtungswechsel = negatives Skalarprodukt
    n_changes = np.sum(dot_products < 0)
    change_rate = n_changes / (len(trajectory) - 2)

    return int(n_changes), float(change_rate)


def compute_self_crossing_index(trajectory, threshold=0.5):
    """
    Misst wie oft die Trajektorie sich selbst nahe kommt (fast-Kreuzungen).

    Returns:
        n_near_crossings: Anzahl Punkte die anderen nahe sind (ohne Nachbarn)
        crossing_density: Dichte der Kreuzungen
    """
    if len(trajectory) < 10:
        return 0, 0.0

    # Paarweise Distanzen
    distances = squareform(pdist(trajectory))

    # Entferne diagonale und nahe Nachbarn
    n = len(trajectory)
    near_crossings = 0

    for i in range(n):
        for j in range(i + 5, n):  # Mindestens 5 Schritte Abstand
            if distances[i, j] < threshold:
                near_crossings += 1

    max_possible = (n * (n - 10)) // 2 if n > 10 else 1
    crossing_density = near_crossings / max_possible

    return int(near_crossings), float(crossing_density)


def estimate_fractal_dimension(trajectory, n_scales=10):
    """
    Schaetzt fraktale Dimension via Box-counting Approximation.

    Returns:
        dim: Geschaetzte Dimension
    """
    if len(trajectory) < 10:
        return 1.0

    # Normalisiere auf Einheits-Hyperkubus
    traj_norm = trajectory - trajectory.min(axis=0)
    max_range = traj_norm.max() + 1e-10
    traj_norm = traj_norm / max_range

    # Box-counting bei verschiedenen Skalen
    counts = []
    scales = []

    for i in range(1, n_scales + 1):
        # Anzahl Boxen pro Dimension
        n_boxes = 2 ** i
        box_size = 1.0 / n_boxes

        # Diskretisiere Punkte in Boxen
        box_indices = (traj_norm // box_size).astype(int)
        box_indices = np.clip(box_indices, 0, n_boxes - 1)

        # Unique Boxen
        unique_boxes = len(set(tuple(b) for b in box_indices))

        counts.append(unique_boxes)
        scales.append(1.0 / n_boxes)

    # Lineare Regression in log-log
    log_scales = np.log(scales)
    log_counts = np.log(counts)

    # Steigung = fraktale Dimension
    slope, _ = np.polyfit(log_scales, log_counts, 1)

    return float(-slope)


def compute_trajectory_complexity(trajectory, threshold=0.5):
    """
    Kombinierte Komplexitaets-Metrik fuer eine Trajektorie.

    Returns:
        complexity: Dict mit verschiedenen Metriken
    """
    n_changes, change_rate = count_direction_changes(trajectory)
    n_crossings, crossing_density = compute_self_crossing_index(trajectory, threshold)
    fractal_dim = estimate_fractal_dimension(trajectory)

    # Gesamtlaenge der Trajektorie
    path_length = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))

    # Direkter Abstand Start -> Ende
    displacement = np.linalg.norm(trajectory[-1] - trajectory[0])

    # Tortuositaet = Pfadlaenge / Displacement
    tortuosity = path_length / (displacement + 1e-10)

    return {
        'n_direction_changes': n_changes,
        'direction_change_rate': change_rate,
        'n_near_crossings': n_crossings,
        'crossing_density': crossing_density,
        'fractal_dimension': fractal_dim,
        'path_length': float(path_length),
        'displacement': float(displacement),
        'tortuosity': float(tortuosity)
    }


def compute_persistence_gudhi(trajectory, max_dimension=1, max_edge=2.0):
    """
    Berechnet persistente Homologie mit gudhi (wenn verfuegbar).

    Returns:
        persistence: Liste von (dimension, birth, death)
        betti: Betti-Zahlen
    """
    if not GUDHI_AVAILABLE:
        return [], {}

    # Sample Trajektorie wenn zu lang
    if len(trajectory) > 500:
        indices = np.linspace(0, len(trajectory) - 1, 500, dtype=int)
        trajectory = trajectory[indices]

    # Rips-Komplex
    rips = gudhi.RipsComplex(points=trajectory, max_edge_length=max_edge)
    simplex_tree = rips.create_simplex_tree(max_dimension=max_dimension + 1)

    # Persistenz berechnen
    simplex_tree.compute_persistence()
    persistence = simplex_tree.persistence()

    # Filtere unendliche Persistenzen
    finite_persistence = [
        (dim, birth, death) for dim, (birth, death) in persistence
        if death != float('inf')
    ]

    # Betti-Zahlen
    betti = simplex_tree.betti_numbers()

    return finite_persistence, betti


def analyze_all_words(emb_reader, vocab_data, sample_interval=10, max_steps=None):
    """
    Analysiert topologische Komplexitaet aller Woerter.

    Returns:
        word_complexity: Dict {word: complexity_metrics}
    """
    idx_to_word = vocab_data['idx_to_word']
    vocab_size = len(vocab_data['vocabulary'])

    n = emb_reader.num_snapshots
    if max_steps:
        n = min(n, max_steps)

    word_complexity = {}

    print(f"  Analysiere {vocab_size} Woerter...")

    for word_idx in range(vocab_size):
        word = idx_to_word.get(str(word_idx), f"w{word_idx}")

        # Lade Trajektorie
        trajectory = emb_reader.read_word_trajectory(word_idx, 0, n, sample_interval)

        # Berechne Komplexitaet
        complexity = compute_trajectory_complexity(trajectory)

        # Wenn gudhi verfuegbar, berechne auch Persistenz
        if GUDHI_AVAILABLE:
            persistence, betti = compute_persistence_gudhi(trajectory)
            complexity['persistence_pairs'] = len(persistence)
            complexity['betti_0'] = betti.get(0, 0)
            complexity['betti_1'] = betti.get(1, 0)

        word_complexity[word] = complexity

        if (word_idx + 1) % 10 == 0:
            print(f"    {word_idx + 1} / {vocab_size}")

    return word_complexity


def compute_topological_similarity(word_complexity):
    """
    Berechnet topologische Aehnlichkeit zwischen Woertern.

    Returns:
        similarity_matrix: (vocab_size, vocab_size) Matrix
        clusters: Cluster von aehnlichen Woertern
    """
    words = list(word_complexity.keys())
    n_words = len(words)

    # Feature-Vektor pro Wort
    features = []
    for word in words:
        c = word_complexity[word]
        feat = [
            c['direction_change_rate'],
            c['crossing_density'],
            c['fractal_dimension'],
            c['tortuosity']
        ]
        features.append(feat)

    features = np.array(features)

    # Normalisiere
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-10)

    # Paarweise Distanzen
    distances = squareform(pdist(features))

    # Similarity = 1 / (1 + distance)
    similarity = 1 / (1 + distances)

    # Einfaches Clustering: Finde aehnlichste Paare
    clusters = []
    for i in range(n_words):
        most_similar = np.argsort(similarity[i])[::-1][1:4]  # Top 3 ohne sich selbst
        clusters.append({
            'word': words[i],
            'similar_to': [words[j] for j in most_similar],
            'similarities': [float(similarity[i, j]) for j in most_similar]
        })

    return similarity, clusters


def visualize_results(word_complexity, similarity_matrix, clusters, vocab_data, output_dir):
    """Erstellt Visualisierungen."""

    words = list(word_complexity.keys())
    word_counts = vocab_data.get('word_counts', {})

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Komplexitaets-Verteilung
    ax = axes[0, 0]
    metrics = ['direction_change_rate', 'crossing_density', 'fractal_dimension', 'tortuosity']
    for metric in metrics:
        values = [word_complexity[w][metric] for w in words]
        ax.hist(values, bins=20, alpha=0.5, label=metric)
    ax.set_xlabel('Wert')
    ax.set_ylabel('Haeufigkeit')
    ax.set_title('Verteilung der Komplexitaets-Metriken')
    ax.legend(fontsize=8)

    # 2. Frequenz vs Komplexitaet
    ax = axes[0, 1]
    freqs = [word_counts.get(w, 1) for w in words]
    complexities = [word_complexity[w]['tortuosity'] for w in words]
    ax.scatter(freqs, complexities, alpha=0.7)
    for i, word in enumerate(words[:10]):
        ax.annotate(word, (freqs[i], complexities[i]), fontsize=7)
    ax.set_xlabel('Wort-Frequenz')
    ax.set_ylabel('Tortuositaet')
    ax.set_title('Frequenz vs Trajektorie-Komplexitaet')

    # 3. Similarity Matrix
    ax = axes[0, 2]
    im = ax.imshow(similarity_matrix, cmap='viridis', aspect='auto')
    ax.set_xlabel('Wort Index')
    ax.set_ylabel('Wort Index')
    ax.set_title('Topologische Aehnlichkeit')
    plt.colorbar(im, ax=ax)

    # 4. Richtungswechsel vs Kreuzungen
    ax = axes[1, 0]
    changes = [word_complexity[w]['direction_change_rate'] for w in words]
    crossings = [word_complexity[w]['crossing_density'] for w in words]
    ax.scatter(changes, crossings, alpha=0.7)
    for i, word in enumerate(words[:10]):
        ax.annotate(word, (changes[i], crossings[i]), fontsize=7)
    ax.set_xlabel('Direction Change Rate')
    ax.set_ylabel('Crossing Density')
    ax.set_title('Richtungswechsel vs Selbst-Kreuzungen')

    # 5. Fraktale Dimension pro Wort
    ax = axes[1, 1]
    fractal_dims = [word_complexity[w]['fractal_dimension'] for w in words]
    sorted_indices = np.argsort(fractal_dims)[::-1]
    sorted_words = [words[i] for i in sorted_indices]
    sorted_dims = [fractal_dims[i] for i in sorted_indices]

    ax.barh(range(len(sorted_words)), sorted_dims, alpha=0.7)
    ax.set_yticks(range(len(sorted_words)))
    ax.set_yticklabels(sorted_words, fontsize=7)
    ax.set_xlabel('Fraktale Dimension')
    ax.set_title('Fraktale Dimension pro Wort')
    ax.invert_yaxis()

    # 6. Zusammenfassung
    ax = axes[1, 2]
    ax.axis('off')

    mean_tort = np.mean([c['tortuosity'] for c in word_complexity.values()])
    mean_frac = np.mean([c['fractal_dimension'] for c in word_complexity.values()])
    max_cross = max([c['crossing_density'] for c in word_complexity.values()])

    # Top aehnliche Paare
    top_similar = []
    for cluster in clusters[:5]:
        top_similar.append(f"{cluster['word']} ~ {cluster['similar_to'][0]}")

    summary = f"""
    TOPOLOGICAL PERSISTENCE ANALYSIS (TPA)
    =====================================

    GUDHI: {'Verfuegbar' if GUDHI_AVAILABLE else 'Nicht verfuegbar (Fallback)'}

    TRAJEKTORIE-KOMPLEXITAET:
    - Woerter analysiert: {len(word_complexity)}
    - Mean Tortuositaet: {mean_tort:.2f}
    - Mean Fraktale Dim: {mean_frac:.2f}
    - Max Crossing Density: {max_cross:.4f}

    TOP KOMPLEXE WOERTER:
    {chr(10).join(f"  {w}: {word_complexity[w]['tortuosity']:.2f}" for w in sorted_words[:5])}

    TOPOLOGISCH AEHNLICHE PAARE:
    {chr(10).join(f"  {pair}" for pair in top_similar)}

    INTERPRETATION:
    - Hohe Tortuositaet = Viele Umwege
    - Hohe Fraktale Dim = Komplex verschlungen
    - Hohe Crossing Density = Oft zum Startpunkt
    - Aehnliche Woerter = Gleiche Trajektorie-Geometrie
    """
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'tpa_analysis.png')
    plt.savefig(output_path, dpi=150)
    print(f"  Gespeichert: {output_path}")
    plt.close()


def run_analysis(max_steps=None, sample_interval=10):
    """
    Fuehrt die vollstaendige TPA-Analyse durch.
    """
    output_dir = ensure_output_dir()

    print("=" * 70)
    print("TOPOLOGICAL PERSISTENCE ANALYSIS (TPA)")
    print("=" * 70)

    print(f"\n  gudhi verfuegbar: {GUDHI_AVAILABLE}")

    vocab_data = load_vocabulary()

    print(f"\n  Konfiguration:")
    print(f"    Sample Interval: {sample_interval}")

    with EmbeddingReader(EMBEDDINGS_FILE) as reader:
        # Analysiere alle Woerter
        print("\n[1/2] Analysiere Wort-Trajektorien...")
        word_complexity = analyze_all_words(reader, vocab_data, sample_interval, max_steps)
        print(f"  {len(word_complexity)} Woerter analysiert")

        # Berechne Aehnlichkeit
        print("\n[2/2] Berechne topologische Aehnlichkeit...")
        similarity_matrix, clusters = compute_topological_similarity(word_complexity)

        # Visualisierungen
        print("\n  Erstelle Visualisierungen...")
        visualize_results(word_complexity, similarity_matrix, clusters, vocab_data, output_dir)

    # Ergebnisse
    results = {
        'gudhi_available': GUDHI_AVAILABLE,
        'n_words': len(word_complexity),
        'mean_tortuosity': float(np.mean([c['tortuosity'] for c in word_complexity.values()])),
        'mean_fractal_dim': float(np.mean([c['fractal_dimension'] for c in word_complexity.values()])),
        'word_complexity': {w: word_complexity[w] for w in list(word_complexity.keys())[:10]},
        'top_similar_pairs': clusters[:5]
    }

    print("\n" + "=" * 70)
    print("TPA ANALYSE ABGESCHLOSSEN")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_analysis()
