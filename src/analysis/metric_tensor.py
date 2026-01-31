"""
Methode 13: Metric Tensor Evolution (MTE)
========================================

Konzept: Gedaechtnis als Aenderung der Distanzfunktion selbst.
Was "weit" war wird "nah" und umgekehrt. Die "Form des Raumes" evolviert.

Ansatz:
- Metrik aus Change-Vektoren schaetzen (Outer-Product)
- Paarweise Distanzen zwischen Woertern tracken
- Aenderungsrate der Distanzen als Metrik-Evolution

Frage beantwortet:
- Wie aendert sich die "Form des Raumes"?
- Wann reorganisiert sich die Semantik?

Autor: Maximilian (MI-Forschung)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import os

from .common import StepReader, EmbeddingReader, STEPS_FILE, EMBEDDINGS_FILE, load_vocabulary
from .common.utils import ensure_output_dir


def compute_metric_from_changes(changes, epsilon=1e-10):
    """
    Schaetzt lokale Metrik aus Change-Vektoren.
    Metrik = durchschnittliches Outer-Product der normalisierten Changes.

    Returns:
        metric: (10, 10) Matrix
        eigenvalues: Sortierte Eigenwerte
    """
    # Normalisiere Changes
    norms = np.linalg.norm(changes, axis=1, keepdims=True)
    normalized = changes / (norms + epsilon)

    # Outer Products
    outer_products = np.array([np.outer(c, c) for c in normalized])

    # Durchschnitt
    metric = np.mean(outer_products, axis=0)

    # Eigenwerte
    eigenvalues = np.linalg.eigvalsh(metric)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Absteigend

    return metric, eigenvalues


def compute_distance_matrix(embedding_snapshot):
    """
    Berechnet paarweise Distanzen zwischen allen Woertern.

    Returns:
        distances: (vocab_size, vocab_size) Matrix
    """
    distances = squareform(pdist(embedding_snapshot))
    return distances


def track_metric_evolution(reader, window_size=1000, stride=1000, max_steps=None):
    """
    Trackt wie sich die Metrik ueber die Zeit aendert.

    Returns:
        metrics: Liste von {step, metric, eigenvalues}
        eigenvalue_ratios: Liste von {step, ratio}
    """
    metrics = []
    eigenvalue_ratios = []

    n = min(reader.num_steps, max_steps) if max_steps else reader.num_steps

    print(f"  Analysiere {n // stride} Fenster...")

    for start in range(0, n - window_size, stride):
        # Lade Changes fuer dieses Fenster
        changes = []
        for i in range(start, start + window_size):
            step = reader.read_step(i)
            changes.append(step['change'])
        changes = np.array(changes)

        # Berechne Metrik
        metric, eigenvalues = compute_metric_from_changes(changes)

        step_idx = start + window_size // 2

        # Eigenvalue Ratio (Anisotropie)
        if eigenvalues[-1] > 1e-10:
            ratio = eigenvalues[0] / eigenvalues[-1]
        else:
            ratio = float('inf')

        metrics.append({
            'step': step_idx,
            'eigenvalues': eigenvalues.tolist(),
            'trace': float(np.trace(metric)),
            'determinant': float(np.linalg.det(metric) if np.linalg.det(metric) > 0 else 0)
        })

        eigenvalue_ratios.append({
            'step': step_idx,
            'ratio': float(ratio) if ratio != float('inf') else 1000,
            'max_eigenvalue': float(eigenvalues[0]),
            'min_eigenvalue': float(eigenvalues[-1])
        })

        if len(metrics) % 100 == 0:
            print(f"    {len(metrics)} Fenster analysiert")

    return metrics, eigenvalue_ratios


def track_distance_evolution(emb_reader, sample_interval=1000, max_snapshots=None):
    """
    Trackt wie sich Wort-Distanzen ueber die Zeit aendern.

    Returns:
        distance_history: Liste von {step, distances_flat}
        distance_changes: Liste von {step, mean_change, max_change}
    """
    distance_history = []
    distance_changes = []

    n = emb_reader.num_snapshots
    if max_snapshots:
        n = min(n, max_snapshots)

    prev_distances = None

    print(f"  Analysiere {n // sample_interval} Snapshots...")

    for i in range(0, n, sample_interval):
        snapshot = emb_reader.read_snapshot(i)
        distances = compute_distance_matrix(snapshot)

        distance_history.append({
            'step': i,
            'mean_distance': float(np.mean(distances)),
            'std_distance': float(np.std(distances)),
            'min_distance': float(np.min(distances[distances > 0])) if np.any(distances > 0) else 0,
            'max_distance': float(np.max(distances))
        })

        if prev_distances is not None:
            # Aenderung der Distanzen
            change = np.abs(distances - prev_distances)
            distance_changes.append({
                'step': i,
                'mean_change': float(np.mean(change)),
                'max_change': float(np.max(change)),
                'std_change': float(np.std(change))
            })

        prev_distances = distances.copy()

        if len(distance_history) % 100 == 0:
            print(f"    {len(distance_history)} Snapshots analysiert")

    return distance_history, distance_changes


def find_metric_transitions(eigenvalue_ratios, threshold_percentile=90):
    """
    Findet Momente wo sich die Metrik stark aendert.

    Returns:
        transitions: Liste von {step, ratio_change}
    """
    transitions = []

    for i in range(1, len(eigenvalue_ratios)):
        prev_ratio = eigenvalue_ratios[i-1]['ratio']
        curr_ratio = eigenvalue_ratios[i]['ratio']

        ratio_change = abs(curr_ratio - prev_ratio)

        transitions.append({
            'step': eigenvalue_ratios[i]['step'],
            'ratio_change': ratio_change,
            'from_ratio': prev_ratio,
            'to_ratio': curr_ratio
        })

    # Finde signifikante Transitionen
    changes = [t['ratio_change'] for t in transitions]
    threshold = np.percentile(changes, threshold_percentile)

    significant = [t for t in transitions if t['ratio_change'] > threshold]
    significant.sort(key=lambda x: x['ratio_change'], reverse=True)

    return significant


def analyze_semantic_reorganization(emb_reader, vocab_data, sample_interval=5000, max_snapshots=None):
    """
    Analysiert ob sich semantische Beziehungen reorganisieren.

    Returns:
        reorganization: Dict mit Metriken
    """
    idx_to_word = vocab_data['idx_to_word']
    n = emb_reader.num_snapshots
    if max_snapshots:
        n = min(n, max_snapshots)

    # Sample Start und Ende
    start_snapshot = emb_reader.read_snapshot(0)
    end_idx = (n // sample_interval - 1) * sample_interval
    end_snapshot = emb_reader.read_snapshot(end_idx) if end_idx > 0 else start_snapshot

    start_distances = compute_distance_matrix(start_snapshot)
    end_distances = compute_distance_matrix(end_snapshot)

    # Finde groesste Distanzaenderungen
    change = end_distances - start_distances
    vocab_size = change.shape[0]

    changes_list = []
    for i in range(vocab_size):
        for j in range(i + 1, vocab_size):
            word_i = idx_to_word.get(str(i), f"w{i}")
            word_j = idx_to_word.get(str(j), f"w{j}")
            changes_list.append({
                'pair': (word_i, word_j),
                'start_distance': float(start_distances[i, j]),
                'end_distance': float(end_distances[i, j]),
                'change': float(change[i, j])
            })

    # Sortiere nach absoluter Aenderung
    changes_list.sort(key=lambda x: abs(x['change']), reverse=True)

    return {
        'top_converging': [c for c in changes_list[:10] if c['change'] < 0],
        'top_diverging': [c for c in changes_list[:10] if c['change'] > 0],
        'mean_abs_change': float(np.mean(np.abs(change))),
        'correlation': float(np.corrcoef(start_distances.flatten(), end_distances.flatten())[0, 1])
    }


def visualize_results(metrics, eigenvalue_ratios, distance_history, distance_changes,
                      transitions, reorganization, output_dir):
    """Erstellt Visualisierungen."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Eigenvalue Ratio ueber Zeit
    ax = axes[0, 0]
    if eigenvalue_ratios:
        steps = [e['step'] for e in eigenvalue_ratios]
        ratios = [min(e['ratio'], 100) for e in eigenvalue_ratios]  # Cap fuer Visualisierung
        ax.plot(steps, ratios, 'b-', alpha=0.7)
        ax.set_yscale('log')
    ax.set_xlabel('Trainingsschritt')
    ax.set_ylabel('Eigenvalue Ratio (log)')
    ax.set_title('Metrik-Anisotropie ueber Zeit')
    ax.grid(True, alpha=0.3)

    # 2. Eigenvalue Spektrum
    ax = axes[0, 1]
    if metrics:
        # Sample einige Metriken
        sample_indices = np.linspace(0, len(metrics) - 1, 10, dtype=int)
        for idx in sample_indices:
            eigenvalues = metrics[idx]['eigenvalues']
            step = metrics[idx]['step']
            ax.plot(range(len(eigenvalues)), eigenvalues, 'o-', alpha=0.5,
                    label=f'Step {step}' if idx in [0, sample_indices[-1]] else None)
    ax.set_xlabel('Eigenwert-Index')
    ax.set_ylabel('Eigenwert')
    ax.set_title('Eigenwert-Spektrum Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Distanz-Evolution
    ax = axes[0, 2]
    if distance_history:
        steps = [d['step'] for d in distance_history]
        mean_dist = [d['mean_distance'] for d in distance_history]
        ax.plot(steps, mean_dist, 'b-', alpha=0.7, label='Mean')

        std_dist = [d['std_distance'] for d in distance_history]
        ax.fill_between(steps,
                        [m - s for m, s in zip(mean_dist, std_dist)],
                        [m + s for m, s in zip(mean_dist, std_dist)],
                        alpha=0.3)
    ax.set_xlabel('Trainingsschritt')
    ax.set_ylabel('Paarweise Distanz')
    ax.set_title('Wort-Distanz Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Distanz-Aenderungsrate
    ax = axes[1, 0]
    if distance_changes:
        steps = [d['step'] for d in distance_changes]
        mean_change = [d['mean_change'] for d in distance_changes]
        ax.plot(steps, mean_change, 'b-', alpha=0.7)

        # Markiere Transitions
        for t in transitions[:10]:
            ax.axvline(t['step'], color='r', alpha=0.3, linestyle='--')

    ax.set_xlabel('Trainingsschritt')
    ax.set_ylabel('Mean Distanz-Aenderung')
    ax.set_title('Distanz-Aenderungsrate (rot = Metrik-Transitions)')
    ax.grid(True, alpha=0.3)

    # 5. Top konvergierende/divergierende Paare
    ax = axes[1, 1]
    if reorganization:
        converging = reorganization['top_converging'][:5]
        diverging = reorganization['top_diverging'][:5]

        labels = [f"{c['pair'][0]}-{c['pair'][1]}" for c in converging + diverging]
        values = [c['change'] for c in converging + diverging]
        colors = ['green'] * len(converging) + ['red'] * len(diverging)

        if labels:
            y_pos = np.arange(len(labels))
            ax.barh(y_pos, values, color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=8)
            ax.axvline(0, color='black', linestyle='-')

    ax.set_xlabel('Distanz-Aenderung')
    ax.set_title('Semantische Reorganisation (gruen=konvergiert, rot=divergiert)')

    # 6. Zusammenfassung
    ax = axes[1, 2]
    ax.axis('off')

    if metrics and eigenvalue_ratios:
        mean_ratio = np.mean([e['ratio'] for e in eigenvalue_ratios if e['ratio'] < 1000])
    else:
        mean_ratio = 0

    summary = f"""
    METRIC TENSOR EVOLUTION (MTE)
    ============================

    METRIK-EVOLUTION:
    - Fenster analysiert: {len(metrics)}
    - Mean Eigenvalue Ratio: {mean_ratio:.2f}
    - Metrik-Transitions: {len(transitions)}

    DISTANZ-EVOLUTION:
    - Snapshots analysiert: {len(distance_history)}
    - Mean Distanz-Aenderung: {np.mean([d['mean_change'] for d in distance_changes]):.4f} if distance_changes else 0

    SEMANTISCHE REORGANISATION:
    - Start-Ende Korrelation: {reorganization.get('correlation', 0):.3f}
    - Mean abs. Distanzaenderung: {reorganization.get('mean_abs_change', 0):.3f}

    TOP KONVERGIERENDE PAARE:
    {chr(10).join(f"  {c['pair'][0]}-{c['pair'][1]}: {c['change']:.3f}" for c in reorganization.get('top_converging', [])[:3])}

    TOP DIVERGIERENDE PAARE:
    {chr(10).join(f"  {c['pair'][0]}-{c['pair'][1]}: {c['change']:.3f}" for c in reorganization.get('top_diverging', [])[:3])}

    INTERPRETATION:
    - Hohe Anisotropie = Bevorzugte Richtungen
    - Transitions = Metrik-Phasenuebergaenge
    - Konvergenz = Woerter werden aehnlicher
    """
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'mte_analysis.png')
    plt.savefig(output_path, dpi=150)
    print(f"  Gespeichert: {output_path}")
    plt.close()


def run_analysis(max_steps=None, window_size=1000, sample_interval=1000):
    """
    Fuehrt die vollstaendige MTE-Analyse durch.

    Args:
        max_steps: Maximale Schritte (None = alle)
        window_size: Fenstergroesse fuer Metrik-Berechnung
        sample_interval: Sampling-Rate fuer Distanz-Tracking
    """
    output_dir = ensure_output_dir()

    print("=" * 70)
    print("METRIC TENSOR EVOLUTION (MTE)")
    print("=" * 70)

    vocab_data = load_vocabulary()

    print(f"\n  Konfiguration:")
    print(f"    Window Size: {window_size}")
    print(f"    Sample Interval: {sample_interval}")

    # Phase 1: Metrik-Evolution aus Changes
    print("\n[1/3] Analysiere Metrik-Evolution...")
    with StepReader(STEPS_FILE) as reader:
        metrics, eigenvalue_ratios = track_metric_evolution(
            reader, window_size, window_size, max_steps
        )
    print(f"  {len(metrics)} Metriken berechnet")

    # Finde Transitions
    transitions = find_metric_transitions(eigenvalue_ratios)
    print(f"  {len(transitions)} signifikante Transitions gefunden")

    # Phase 2: Distanz-Evolution aus Embeddings
    print("\n[2/3] Analysiere Distanz-Evolution...")
    with EmbeddingReader(EMBEDDINGS_FILE) as reader:
        distance_history, distance_changes = track_distance_evolution(
            reader, sample_interval, max_steps
        )
    print(f"  {len(distance_history)} Snapshots analysiert")

    # Phase 3: Semantische Reorganisation
    print("\n[3/3] Analysiere semantische Reorganisation...")
    with EmbeddingReader(EMBEDDINGS_FILE) as reader:
        reorganization = analyze_semantic_reorganization(
            reader, vocab_data, sample_interval * 5, max_steps
        )
    print(f"  Korrelation Start-Ende: {reorganization['correlation']:.3f}")

    # Visualisierungen
    print("\n  Erstelle Visualisierungen...")
    visualize_results(metrics, eigenvalue_ratios, distance_history, distance_changes,
                      transitions, reorganization, output_dir)

    # Ergebnisse
    results = {
        'n_metrics': len(metrics),
        'n_transitions': len(transitions),
        'mean_eigenvalue_ratio': float(np.mean([e['ratio'] for e in eigenvalue_ratios if e['ratio'] < 1000])) if eigenvalue_ratios else 0,
        'top_transitions': transitions[:5],
        'distance_evolution': {
            'n_snapshots': len(distance_history),
            'mean_distance_change': float(np.mean([d['mean_change'] for d in distance_changes])) if distance_changes else 0
        },
        'reorganization': reorganization
    }

    print("\n" + "=" * 70)
    print("MTE ANALYSE ABGESCHLOSSEN")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_analysis()
