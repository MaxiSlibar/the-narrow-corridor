"""
Methode 10: Forbidden State Sequence Analysis (FSSA)
====================================================

Konzept: Analysiere was NICHT passiert - welche Embedding-Konfigurationen
werden nach bestimmten Trainingsschritten UNERREICHBAR?

Ansatz:
- System-Zustand = k-means Cluster-ID (k=100) statt 8^34 Oktanten
- Zustandsuebergangsgraph (100x100 Matrix) bauen
- "Closing Events" finden: Schritte wo besuchte Cluster-Anzahl SINKT

Frage beantwortet:
- Wann "committed" das Training zu bestimmten Pfaden?
- Gibt es "Points of no return"?

Autor: Maximilian (MI-Forschung)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import defaultdict
import os

from .common import EmbeddingReader, EMBEDDINGS_FILE, load_vocabulary
from .common.utils import ensure_output_dir


def sample_embeddings_for_clustering(emb_reader, sample_every=1000, max_samples=1000):
    """
    Sampled Embeddings fuer k-means Clustering.

    Returns:
        samples: Array (n_samples, vocab_size * embedding_dim)
        indices: Liste der Snapshot-Indizes
    """
    samples = []
    indices = []

    n_snapshots = emb_reader.num_snapshots
    step = max(1, n_snapshots // max_samples)

    print(f"  Sampling Embeddings alle {step} Schritte...")

    for i in range(0, n_snapshots, step):
        if len(samples) >= max_samples:
            break
        snapshot = emb_reader.read_snapshot(i)
        samples.append(snapshot.flatten())
        indices.append(i)

        if len(samples) % 100 == 0:
            print(f"    {len(samples)} / {max_samples}")

    return np.array(samples), indices


def fit_state_clusters(samples, n_clusters=100):
    """
    Fittet k-means auf gesampelten Embeddings.

    Returns:
        kmeans: Gefittetes KMeans-Modell
    """
    print(f"  Fitte k-means mit {n_clusters} Clustern auf {len(samples)} Samples...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(samples)
    print(f"  Inertia: {kmeans.inertia_:.2f}")
    return kmeans


def classify_all_snapshots(emb_reader, kmeans, sample_every=100, max_snapshots=None):
    """
    Klassifiziert Snapshots in Cluster-IDs.

    Returns:
        states: Liste von Cluster-IDs
        indices: Liste der Snapshot-Indizes
    """
    states = []
    indices = []

    n = emb_reader.num_snapshots
    if max_snapshots:
        n = min(n, max_snapshots)

    print(f"  Klassifiziere {n // sample_every} Snapshots...")

    for i in range(0, n, sample_every):
        snapshot = emb_reader.read_snapshot(i)
        state = kmeans.predict(snapshot.flatten().reshape(1, -1))[0]
        states.append(state)
        indices.append(i)

        if len(states) % 1000 == 0:
            print(f"    {len(states)} klassifiziert")

    return states, indices


def build_transition_matrix(states, n_clusters):
    """
    Baut Zustandsuebergangsmatrix.

    Returns:
        T: Uebergangsmatrix (n_clusters, n_clusters)
        counts: Rohe Zaehlungen
    """
    counts = np.zeros((n_clusters, n_clusters), dtype=np.int64)

    for i in range(len(states) - 1):
        from_state = states[i]
        to_state = states[i + 1]
        counts[from_state, to_state] += 1

    # Normalisiere zu Wahrscheinlichkeiten
    row_sums = counts.sum(axis=1, keepdims=True)
    T = np.divide(counts, row_sums, where=row_sums > 0)
    T = np.nan_to_num(T)

    return T, counts


def find_forbidden_transitions(counts, min_expected=1):
    """
    Findet Transitionen die NIE auftreten obwohl sie koennten.

    Returns:
        forbidden: Liste von (from_state, to_state) Tupeln
        forbidden_rate: Anteil verbotener Transitionen
    """
    # Zustaende die mindestens einmal besucht wurden
    visited_states = set(np.where(counts.sum(axis=1) > 0)[0])

    forbidden = []
    possible_count = 0

    for from_state in visited_states:
        for to_state in visited_states:
            possible_count += 1
            if counts[from_state, to_state] == 0:
                forbidden.append((from_state, to_state))

    forbidden_rate = len(forbidden) / possible_count if possible_count > 0 else 0

    return forbidden, forbidden_rate


def compute_reachability_over_time(states, indices, window_size=100, n_clusters=100):
    """
    Berechnet wie viele Cluster in einem Zeitfenster erreichbar sind.

    Returns:
        reachability: Liste von (step_idx, n_reachable) Tupeln
        closing_events: Schritte wo Erreichbarkeit sinkt
    """
    reachability = []
    closing_events = []

    prev_reachable = None

    for start in range(0, len(states) - window_size, window_size // 2):
        end = start + window_size
        window_states = states[start:end]

        # Unique Zustaende im Fenster
        unique_states = set(window_states)
        n_reachable = len(unique_states)

        step_idx = indices[start + window_size // 2]
        reachability.append((step_idx, n_reachable))

        # Closing Event = Abnahme der Erreichbarkeit
        if prev_reachable is not None:
            if n_reachable < prev_reachable * 0.8:  # 20% Abnahme
                closing_events.append({
                    'step': step_idx,
                    'from_reachable': prev_reachable,
                    'to_reachable': n_reachable,
                    'drop_ratio': n_reachable / prev_reachable
                })

        prev_reachable = n_reachable

    return reachability, closing_events


def compute_state_entropy(states, n_clusters):
    """
    Berechnet Entropie der Zustandsverteilung.

    Returns:
        entropy: Shannon-Entropie in bits
        max_entropy: Maximale moegliche Entropie
    """
    counts = np.bincount(states, minlength=n_clusters)
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Nur nicht-null

    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(n_clusters)

    return entropy, max_entropy


def analyze_run_differences(states, indices, steps_per_run=21000, n_runs=50):
    """
    Analysiert Unterschiede zwischen Runs.

    Returns:
        run_stats: Dict mit Statistiken pro Run
    """
    run_stats = {}

    for run_id in range(n_runs):
        # Finde Indizes fuer diesen Run
        run_start = run_id * steps_per_run
        run_end = (run_id + 1) * steps_per_run

        # Finde States die zu diesem Run gehoeren
        run_states = []
        for i, idx in enumerate(indices):
            if run_start <= idx < run_end:
                run_states.append(states[i])

        if len(run_states) > 0:
            unique_states = len(set(run_states))
            entropy, _ = compute_state_entropy(run_states, 100)
            run_stats[run_id] = {
                'unique_states': unique_states,
                'entropy': entropy,
                'n_samples': len(run_states)
            }

    return run_stats


def visualize_results(T, counts, reachability, closing_events,
                      forbidden_rate, entropy, run_stats, output_dir):
    """Erstellt Visualisierungen."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Uebergangsmatrix
    ax = axes[0, 0]
    im = ax.imshow(np.log1p(counts), cmap='Blues', aspect='auto')
    ax.set_xlabel('To State')
    ax.set_ylabel('From State')
    ax.set_title('Zustandsuebergangs-Matrix (log-Skala)')
    plt.colorbar(im, ax=ax)

    # 2. Erreichbarkeit ueber Zeit
    ax = axes[0, 1]
    steps, n_reach = zip(*reachability) if reachability else ([], [])
    ax.plot(steps, n_reach, 'b-', alpha=0.7)

    # Closing Events markieren
    for event in closing_events:
        ax.axvline(event['step'], color='r', alpha=0.5, linestyle='--')

    ax.set_xlabel('Trainingsschritt')
    ax.set_ylabel('Erreichbare Zustaende')
    ax.set_title(f'Erreichbarkeit ueber Zeit ({len(closing_events)} Closing Events)')
    ax.grid(True, alpha=0.3)

    # 3. Zustandsverteilung
    ax = axes[0, 2]
    state_counts = np.bincount([s for s in range(100)], minlength=100)
    # Zaehle tatsaechliche Besuche
    for s in range(len(T)):
        state_counts[s] = counts[s].sum()
    ax.bar(range(100), state_counts, alpha=0.7)
    ax.set_xlabel('Zustand (Cluster-ID)')
    ax.set_ylabel('Besuchshaeufigkeit')
    ax.set_title('Zustandsverteilung')

    # 4. Run-Vergleich
    ax = axes[1, 0]
    if run_stats:
        runs = sorted(run_stats.keys())
        unique_per_run = [run_stats[r]['unique_states'] for r in runs]
        ax.bar(runs, unique_per_run, alpha=0.7)
        ax.set_xlabel('Run ID')
        ax.set_ylabel('Unique Zustaende')
        ax.set_title('Zustandsdiversitaet pro Run')

    # 5. Verbotene Transitionen
    ax = axes[1, 1]
    # Binarisiere: 0 = verboten, 1 = erlaubt
    forbidden_matrix = (counts == 0).astype(float)
    # Nur fuer besuchte Zustaende
    visited = counts.sum(axis=1) > 0
    forbidden_matrix[~visited, :] = np.nan
    forbidden_matrix[:, ~visited] = np.nan

    im = ax.imshow(forbidden_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    ax.set_xlabel('To State')
    ax.set_ylabel('From State')
    ax.set_title(f'Verbotene Transitionen (rot) - {forbidden_rate*100:.1f}%')
    plt.colorbar(im, ax=ax)

    # 6. Zusammenfassung
    ax = axes[1, 2]
    ax.axis('off')

    summary = f"""
    FORBIDDEN STATE SEQUENCE ANALYSIS (FSSA)
    ========================================

    ZUSTANDSRAUM:
    - 100 Cluster (k-means)
    - Besuchte Zustaende: {(counts.sum(axis=1) > 0).sum()}
    - Shannon-Entropie: {entropy[0]:.2f} / {entropy[1]:.2f} bits

    TRANSITIONEN:
    - Beobachtete Transitionen: {(counts > 0).sum()}
    - Verbotene Transitionen: {forbidden_rate*100:.1f}%

    CLOSING EVENTS:
    - Anzahl: {len(closing_events)}
    - Durchschn. Drop: {np.mean([e['drop_ratio'] for e in closing_events])*100:.1f}%
      (wenn Events vorhanden)

    RUN-VERGLEICH:
    - Runs analysiert: {len(run_stats)}
    - Min Unique States: {min(r['unique_states'] for r in run_stats.values()) if run_stats else 'N/A'}
    - Max Unique States: {max(r['unique_states'] for r in run_stats.values()) if run_stats else 'N/A'}

    INTERPRETATION:
    - Hohe forbidden_rate = System stark eingeschraenkt
    - Closing Events = "Points of no return"
    - Run-Varianz = Verschiedene Pfade moeglich
    """
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'fssa_analysis.png')
    plt.savefig(output_path, dpi=150)
    print(f"  Gespeichert: {output_path}")
    plt.close()


def run_analysis(max_steps=None, n_clusters=100, sample_every_classify=100):
    """
    Fuehrt die vollstaendige FSSA-Analyse durch.

    Args:
        max_steps: Maximale Schritte (None = alle)
        n_clusters: Anzahl k-means Cluster
        sample_every_classify: Sampling-Rate fuer Klassifikation
    """
    output_dir = ensure_output_dir()

    print("=" * 70)
    print("FORBIDDEN STATE SEQUENCE ANALYSIS (FSSA)")
    print("=" * 70)

    vocab_data = load_vocabulary()
    config = vocab_data['config']
    steps_per_run = config['num_pairs'] * config['epochs']

    print(f"\n  Konfiguration:")
    print(f"    Cluster: {n_clusters}")
    print(f"    Schritte pro Run: {steps_per_run}")

    with EmbeddingReader(EMBEDDINGS_FILE) as reader:
        # 1. Sample fuer Clustering
        print("\n[1/5] Sampling fuer Clustering...")
        samples, sample_indices = sample_embeddings_for_clustering(
            reader, sample_every=1000, max_samples=1000
        )

        # 2. Fit k-means
        print("\n[2/5] K-Means Clustering...")
        kmeans = fit_state_clusters(samples, n_clusters=n_clusters)

        # 3. Klassifiziere alle Snapshots
        print("\n[3/5] Klassifiziere Snapshots...")
        states, indices = classify_all_snapshots(
            reader, kmeans,
            sample_every=sample_every_classify,
            max_snapshots=max_steps
        )
        print(f"  {len(states)} Snapshots klassifiziert")

        # 4. Baue Uebergangsmatrix
        print("\n[4/5] Baue Uebergangsmatrix...")
        T, counts = build_transition_matrix(states, n_clusters)
        print(f"  Non-zero Transitionen: {(counts > 0).sum()}")

        # 5. Analysiere
        print("\n[5/5] Analysiere...")

        # Verbotene Transitionen
        forbidden, forbidden_rate = find_forbidden_transitions(counts)
        print(f"  Verbotene Transitionen: {len(forbidden)} ({forbidden_rate*100:.1f}%)")

        # Erreichbarkeit
        reachability, closing_events = compute_reachability_over_time(
            states, indices, window_size=100, n_clusters=n_clusters
        )
        print(f"  Closing Events: {len(closing_events)}")

        # Entropie
        entropy = compute_state_entropy(states, n_clusters)
        print(f"  Entropie: {entropy[0]:.2f} / {entropy[1]:.2f} bits")

        # Run-Vergleich
        run_stats = analyze_run_differences(states, indices, steps_per_run)
        print(f"  Runs analysiert: {len(run_stats)}")

        # Visualisierungen
        print("\n  Erstelle Visualisierungen...")
        visualize_results(T, counts, reachability, closing_events,
                          forbidden_rate, entropy, run_stats, output_dir)

    # Ergebnisse
    results = {
        'n_clusters': n_clusters,
        'n_states_classified': len(states),
        'n_visited_states': int((counts.sum(axis=1) > 0).sum()),
        'n_transitions_observed': int((counts > 0).sum()),
        'forbidden_rate': float(forbidden_rate),
        'n_closing_events': len(closing_events),
        'entropy': float(entropy[0]),
        'max_entropy': float(entropy[1]),
        'closing_events': closing_events[:10],  # Top 10
        'run_stats_summary': {
            'n_runs': len(run_stats),
            'mean_unique_states': float(np.mean([r['unique_states'] for r in run_stats.values()])) if run_stats else 0,
            'std_unique_states': float(np.std([r['unique_states'] for r in run_stats.values()])) if run_stats else 0
        }
    }

    print("\n" + "=" * 70)
    print("FSSA ANALYSE ABGESCHLOSSEN")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_analysis()
