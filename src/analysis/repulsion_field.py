"""
Methode 15: Repulsion Field Analysis (RFA)
==========================================

Konzept: Finde Zustaende die "abstossen" - Konfigurationen die das System
schnell verlaesst. Diese repulsiven Zonen verraten was Training "vermeidet".

Ansatz:
- "Dwell Time" in verschiedenen Regionen messen
- Repulsions-Vektor berechnen (durchschnittliche Fluchtrichtung)
- Repulsionsfeld im Embedding-Raum kartieren

Frage beantwortet:
- Welche Embedding-Regionen vermeidet das Training?
- Welche impliziten Constraints werden gelernt?

Autor: Maximilian (MI-Forschung)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import defaultdict
import os

from .common import EmbeddingReader, StepReader, EMBEDDINGS_FILE, STEPS_FILE, load_vocabulary
from .common.utils import ensure_output_dir


def discretize_space(embeddings_sample, n_regions=50):
    """
    Diskretisiert den Embedding-Raum in Regionen via k-means.

    Returns:
        kmeans: Gefittetes Modell
        centers: Cluster-Zentren
    """
    print(f"  Fitte k-means mit {n_regions} Regionen...")
    kmeans = KMeans(n_clusters=n_regions, random_state=42, n_init=10)
    kmeans.fit(embeddings_sample)
    return kmeans, kmeans.cluster_centers_


def compute_dwell_times(emb_reader, kmeans, sample_interval=100, max_snapshots=None):
    """
    Berechnet wie lange das System in jeder Region verweilt.

    Returns:
        dwell_times: Dict {region_id: [list of dwell durations]}
        transitions: Liste von (from_region, to_region, step)
    """
    n = emb_reader.num_snapshots
    if max_snapshots:
        n = min(n, max_snapshots)

    dwell_times = defaultdict(list)
    transitions = []

    prev_region = None
    current_dwell = 0
    current_region_start = 0

    print(f"  Analysiere {n // sample_interval} Snapshots...")

    for i in range(0, n, sample_interval):
        snapshot = emb_reader.read_snapshot(i)
        # Gesamten Snapshot als einen Punkt (flatten)
        flat = snapshot.flatten().reshape(1, -1)
        region = kmeans.predict(flat)[0]

        if prev_region is None:
            prev_region = region
            current_dwell = 1
            current_region_start = i
        elif region == prev_region:
            current_dwell += 1
        else:
            # Region gewechselt
            dwell_times[prev_region].append({
                'duration': current_dwell,
                'start': current_region_start,
                'end': i
            })
            transitions.append({
                'from': prev_region,
                'to': region,
                'step': i
            })
            prev_region = region
            current_dwell = 1
            current_region_start = i

        if (i // sample_interval) % 500 == 0:
            print(f"    Schritt {i:,}")

    # Letzte Dwell-Zeit
    if current_dwell > 0:
        dwell_times[prev_region].append({
            'duration': current_dwell,
            'start': current_region_start,
            'end': n
        })

    return dict(dwell_times), transitions


def compute_repulsion_vectors(step_reader, emb_reader, kmeans, vocab_data,
                               sample_interval=100, max_steps=None):
    """
    Berechnet Repulsions-Vektoren fuer jede Region.
    Repulsion = durchschnittliche Bewegungsrichtung BEIM VERLASSEN einer Region.

    Returns:
        repulsion: Dict {region_id: repulsion_vector}
    """
    n = min(step_reader.num_steps, max_steps) if max_steps else step_reader.num_steps

    # Sammle Bewegungen beim Verlassen jeder Region
    exit_movements = defaultdict(list)

    prev_region = None
    prev_snapshot_flat = None

    print(f"  Berechne Repulsions-Vektoren...")

    for i in range(0, n, sample_interval):
        snapshot = emb_reader.read_snapshot(i)
        flat = snapshot.flatten()
        region = kmeans.predict(flat.reshape(1, -1))[0]

        if prev_region is not None and region != prev_region:
            # Verlassen einer Region - speichere Bewegung
            movement = flat - prev_snapshot_flat
            exit_movements[prev_region].append(movement)

        prev_region = region
        prev_snapshot_flat = flat.copy()

        if (i // sample_interval) % 500 == 0 and i > 0:
            print(f"    Schritt {i:,}")

    # Berechne durchschnittliche Repulsion pro Region
    repulsion = {}
    for region_id, movements in exit_movements.items():
        if len(movements) > 0:
            avg_movement = np.mean(movements, axis=0)
            repulsion[region_id] = {
                'vector': avg_movement.tolist(),
                'magnitude': float(np.linalg.norm(avg_movement)),
                'n_exits': len(movements)
            }

    return repulsion


def identify_repulsive_regions(dwell_times, repulsion, threshold_percentile=25):
    """
    Identifiziert Regionen die besonders abstossend sind.
    Kriterien: Kurze Dwell-Zeit + Hohe Repulsions-Magnitude.

    Returns:
        repulsive: Liste von {region_id, dwell_score, repulsion_score}
    """
    # Berechne durchschnittliche Dwell-Zeit pro Region
    avg_dwell = {}
    for region_id, dwells in dwell_times.items():
        if dwells:
            avg_dwell[region_id] = np.mean([d['duration'] for d in dwells])

    if not avg_dwell:
        return []

    dwell_threshold = np.percentile(list(avg_dwell.values()), threshold_percentile)

    repulsive = []
    for region_id, avg in avg_dwell.items():
        if avg < dwell_threshold and region_id in repulsion:
            rep_mag = repulsion[region_id]['magnitude']
            repulsive.append({
                'region_id': region_id,
                'avg_dwell': float(avg),
                'repulsion_magnitude': float(rep_mag),
                'n_visits': len(dwell_times[region_id]),
                'n_exits': repulsion[region_id]['n_exits'],
                'combined_score': float(rep_mag / (avg + 0.1))
            })

    repulsive.sort(key=lambda x: x['combined_score'], reverse=True)
    return repulsive


def analyze_repulsion_by_word(step_reader, vocab_data, max_steps=None):
    """
    Analysiert welche Woerter die staerkste "Flucht" zeigen.

    Returns:
        word_repulsion: Dict {word: repulsion_stats}
    """
    idx_to_word = vocab_data['idx_to_word']
    n = min(step_reader.num_steps, max_steps) if max_steps else step_reader.num_steps

    # Sammle Changes pro Wort
    word_changes = defaultdict(list)

    print(f"  Analysiere Wort-Repulsion...")

    for i in range(n):
        step = step_reader.read_step(i)
        target_idx = step['target_idx']
        word = idx_to_word.get(str(target_idx), f"w{target_idx}")
        word_changes[word].append(step['change'])

        if (i + 1) % 100000 == 0:
            print(f"    {i + 1:,} / {n:,}")

    # Berechne Repulsions-Statistiken pro Wort
    word_repulsion = {}
    for word, changes in word_changes.items():
        changes = np.array(changes)
        mean_change = np.mean(changes, axis=0)
        mean_magnitude = np.mean(np.linalg.norm(changes, axis=1))

        # "Konsistenz" der Bewegung (hohe Konsistenz = immer gleiche Richtung)
        if len(changes) > 1:
            consistency = np.linalg.norm(mean_change) / (mean_magnitude + 1e-10)
        else:
            consistency = 0

        word_repulsion[word] = {
            'mean_magnitude': float(mean_magnitude),
            'consistency': float(consistency),
            'n_updates': len(changes)
        }

    return word_repulsion


def visualize_results(dwell_times, repulsion, repulsive_regions, word_repulsion,
                      kmeans_centers, transitions, output_dir):
    """Erstellt Visualisierungen."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Dwell-Zeit Verteilung
    ax = axes[0, 0]
    all_dwells = []
    for region_id, dwells in dwell_times.items():
        for d in dwells:
            all_dwells.append(d['duration'])
    if all_dwells:
        ax.hist(all_dwells, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Dwell-Zeit (Samples)')
    ax.set_ylabel('Haeufigkeit')
    ax.set_title('Verteilung der Verweildauern')

    # 2. Repulsions-Magnitudes pro Region
    ax = axes[0, 1]
    if repulsion:
        regions = sorted(repulsion.keys())
        magnitudes = [repulsion[r]['magnitude'] for r in regions]
        ax.bar(regions, magnitudes, alpha=0.7)

        # Markiere repulsive Regionen
        for rep in repulsive_regions[:5]:
            if rep['region_id'] in regions:
                idx = regions.index(rep['region_id'])
                ax.bar(idx, magnitudes[idx], color='red', alpha=0.7)

    ax.set_xlabel('Region ID')
    ax.set_ylabel('Repulsions-Magnitude')
    ax.set_title('Repulsion pro Region (rot = stark repulsiv)')

    # 3. Wort-Repulsion (Konsistenz vs Magnitude)
    ax = axes[0, 2]
    if word_repulsion:
        words = list(word_repulsion.keys())
        mags = [word_repulsion[w]['mean_magnitude'] for w in words]
        cons = [word_repulsion[w]['consistency'] for w in words]
        ax.scatter(mags, cons, alpha=0.7)

        # Label einige Woerter
        for i, word in enumerate(words[:10]):
            ax.annotate(word, (mags[i], cons[i]), fontsize=8)

    ax.set_xlabel('Mean Magnitude')
    ax.set_ylabel('Konsistenz')
    ax.set_title('Wort-Bewegung (hoch rechts = konsistente Flucht)')

    # 4. Transitions-Matrix
    ax = axes[1, 0]
    if transitions:
        n_regions = len(kmeans_centers)
        trans_matrix = np.zeros((n_regions, n_regions))
        for t in transitions:
            trans_matrix[t['from'], t['to']] += 1

        im = ax.imshow(np.log1p(trans_matrix), cmap='Blues', aspect='auto')
        ax.set_xlabel('To Region')
        ax.set_ylabel('From Region')
        ax.set_title('Region-Transitionen (log)')
        plt.colorbar(im, ax=ax)

    # 5. Repulsive Regionen Ranking
    ax = axes[1, 1]
    if repulsive_regions:
        top = repulsive_regions[:15]
        regions = [r['region_id'] for r in top]
        scores = [r['combined_score'] for r in top]
        ax.barh(range(len(regions)), scores, alpha=0.7)
        ax.set_yticks(range(len(regions)))
        ax.set_yticklabels([f"Region {r}" for r in regions])
        ax.set_xlabel('Combined Repulsion Score')
        ax.set_title('Top Repulsive Regionen')
        ax.invert_yaxis()

    # 6. Zusammenfassung
    ax = axes[1, 2]
    ax.axis('off')

    n_regions_visited = len(dwell_times)
    n_repulsive = len(repulsive_regions)
    mean_dwell = np.mean(all_dwells) if all_dwells else 0

    min_dwell = min(all_dwells) if all_dwells else 0
    max_dwell = max(all_dwells) if all_dwells else 0
    top_region = repulsive_regions[0]['region_id'] if repulsive_regions else 'N/A'
    top_score = repulsive_regions[0]['combined_score'] if repulsive_regions else 0
    max_consistency = max(w['consistency'] for w in word_repulsion.values()) if word_repulsion else 0

    summary = f"""
    REPULSION FIELD ANALYSIS (RFA)
    =============================

    RAUM-DISKRETISIERUNG:
    - Regionen (Cluster): {len(kmeans_centers)}
    - Besuchte Regionen: {n_regions_visited}
    - Transitionen: {len(transitions)}

    DWELL-TIME:
    - Mean Dwell-Zeit: {mean_dwell:.1f} Samples
    - Kuerzeste: {min_dwell:.0f}
    - Laengste: {max_dwell:.0f}

    REPULSIVE REGIONEN:
    - Stark repulsiv: {n_repulsive}
    - Top Region: {top_region}
    - Top Score: {top_score:.2f}

    WORT-ANALYSE:
    - Woerter analysiert: {len(word_repulsion)}
    - Max Konsistenz: {max_consistency:.3f}

    INTERPRETATION:
    - Kurze Dwell + Hohe Repulsion = Region wird gemieden
    - Hohe Konsistenz = Wort bewegt sich immer gleich
    - Viele Transitionen = Instabile Region
    """
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'rfa_analysis.png')
    plt.savefig(output_path, dpi=150)
    print(f"  Gespeichert: {output_path}")
    plt.close()


def run_analysis(max_steps=None, n_regions=50, sample_interval=100):
    """
    Fuehrt die vollstaendige RFA-Analyse durch.
    """
    output_dir = ensure_output_dir()

    print("=" * 70)
    print("REPULSION FIELD ANALYSIS (RFA)")
    print("=" * 70)

    vocab_data = load_vocabulary()

    print(f"\n  Konfiguration:")
    print(f"    Regionen: {n_regions}")
    print(f"    Sample Interval: {sample_interval}")

    with EmbeddingReader(EMBEDDINGS_FILE) as emb_reader:
        # Sample fuer Clustering
        print("\n[1/5] Sample fuer Clustering...")
        n = emb_reader.num_snapshots
        if max_steps:
            n = min(n, max_steps)

        sample_indices = np.linspace(0, n - 1, min(1000, n // 100), dtype=int)
        samples = []
        for idx in sample_indices:
            snapshot = emb_reader.read_snapshot(idx)
            samples.append(snapshot.flatten())
        samples = np.array(samples)
        print(f"  {len(samples)} Samples")

        # Clustering
        print("\n[2/5] Diskretisiere Raum...")
        kmeans, centers = discretize_space(samples, n_regions)

        # Dwell Times
        print("\n[3/5] Berechne Dwell-Times...")
        dwell_times, transitions = compute_dwell_times(
            emb_reader, kmeans, sample_interval, max_steps
        )
        print(f"  {len(dwell_times)} Regionen besucht")
        print(f"  {len(transitions)} Transitionen")

        # Repulsion Vectors
        print("\n[4/5] Berechne Repulsions-Vektoren...")
        with StepReader(STEPS_FILE) as step_reader:
            repulsion = compute_repulsion_vectors(
                step_reader, emb_reader, kmeans, vocab_data,
                sample_interval, max_steps
            )
        print(f"  {len(repulsion)} Regionen mit Repulsion")

        # Repulsive Regions
        repulsive_regions = identify_repulsive_regions(dwell_times, repulsion)
        print(f"  {len(repulsive_regions)} stark repulsive Regionen")

    # Wort-Analyse
    print("\n[5/5] Analysiere Wort-Repulsion...")
    with StepReader(STEPS_FILE) as step_reader:
        word_repulsion = analyze_repulsion_by_word(step_reader, vocab_data, max_steps)

    # Visualisierungen
    print("\n  Erstelle Visualisierungen...")
    visualize_results(dwell_times, repulsion, repulsive_regions, word_repulsion,
                      centers, transitions, output_dir)

    # Ergebnisse
    all_dwells = []
    for dwells in dwell_times.values():
        all_dwells.extend([d['duration'] for d in dwells])

    results = {
        'n_regions': n_regions,
        'n_visited_regions': len(dwell_times),
        'n_transitions': len(transitions),
        'mean_dwell_time': float(np.mean(all_dwells)) if all_dwells else 0,
        'n_repulsive_regions': len(repulsive_regions),
        'top_repulsive': repulsive_regions[:5],
        'word_repulsion_summary': {
            'n_words': len(word_repulsion),
            'max_consistency': float(max(w['consistency'] for w in word_repulsion.values())) if word_repulsion else 0
        }
    }

    print("\n" + "=" * 70)
    print("RFA ANALYSE ABGESCHLOSSEN")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_analysis()
