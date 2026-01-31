"""
Methode 17: Reachability Horizon Collapse (RHC)
==============================================

Konzept: Gedaechtnis als Verschwinden erreichbarer Zustaende messen.
Was wird mit der Zeit UNMOEGLICH? Die Freiheit des Systems schrumpft.

Ansatz:
- Fuer jedes Wort: Historische Change-Verteilung tracken
- Reachability = Standardabweichung der moeglichen Bewegungen
- Collapse = Abnahme der Reachability ueber Zeit

Frage beantwortet:
- Welche Woerter "settlen" frueh vs. spaet?
- Korreliert Frequenz mit Constraint-Akkumulation?

Autor: Maximilian (MI-Forschung)
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

from .common import StepReader, STEPS_FILE, load_vocabulary
from .common.utils import ensure_output_dir


def compute_reachability(changes, method='std'):
    """
    Berechnet "Reachability" als Mass fuer Bewegungsfreiheit.

    Args:
        changes: Array von Change-Vektoren
        method: 'std' (Standardabweichung) oder 'range' (Spannweite)

    Returns:
        reachability: Skalarer Wert
    """
    if len(changes) == 0:
        return 0.0

    changes = np.array(changes)

    if method == 'std':
        # Standardabweichung der Magnitudes
        magnitudes = np.linalg.norm(changes, axis=1)
        return float(np.std(magnitudes))
    elif method == 'range':
        # Spannweite pro Dimension
        ranges = np.ptp(changes, axis=0)
        return float(np.mean(ranges))
    else:
        return 0.0


def track_reachability_per_word(step_reader, vocab_data, checkpoint_interval=1000,
                                 horizon=100, max_steps=None):
    """
    Trackt Reachability pro Wort ueber Zeit.

    Args:
        checkpoint_interval: Alle X Schritte neu berechnen
        horizon: Anzahl Schritte fuer Reachability-Berechnung

    Returns:
        reachability_history: Dict {word: [(step, reachability), ...]}
    """
    idx_to_word = vocab_data['idx_to_word']
    n = min(step_reader.num_steps, max_steps) if max_steps else step_reader.num_steps

    # Sammle alle Steps
    print(f"  Lade Steps...")
    all_steps = []
    for i in range(n):
        step = step_reader.read_step(i)
        all_steps.append({
            'target_idx': step['target_idx'],
            'change': step['change']
        })
        if (i + 1) % 100000 == 0:
            print(f"    {i + 1:,} / {n:,}")

    # Berechne Reachability an Checkpoints
    reachability_history = defaultdict(list)

    print(f"  Berechne Reachability an {n // checkpoint_interval} Checkpoints...")

    for checkpoint in range(0, n - horizon, checkpoint_interval):
        # Sammle Changes der naechsten 'horizon' Schritte pro Wort
        word_changes = defaultdict(list)

        for i in range(checkpoint, min(checkpoint + horizon, n)):
            step = all_steps[i]
            word_idx = step['target_idx']
            word = idx_to_word.get(str(word_idx), f"w{word_idx}")
            word_changes[word].append(step['change'])

        # Berechne Reachability pro Wort
        for word, changes in word_changes.items():
            if len(changes) >= 5:  # Mindestens 5 Samples
                reach = compute_reachability(changes)
                reachability_history[word].append({
                    'step': checkpoint,
                    'reachability': reach,
                    'n_samples': len(changes)
                })

        if (checkpoint // checkpoint_interval) % 100 == 0:
            print(f"    Checkpoint {checkpoint:,}")

    return dict(reachability_history)


def detect_collapse_events(reachability_history, threshold_ratio=0.5):
    """
    Findet Momente wo Reachability stark abfaellt.

    Returns:
        collapse_events: Dict {word: [collapse_events]}
    """
    collapse_events = {}

    for word, history in reachability_history.items():
        if len(history) < 3:
            continue

        events = []
        for i in range(1, len(history)):
            prev_reach = history[i - 1]['reachability']
            curr_reach = history[i]['reachability']

            if prev_reach > 0:
                ratio = curr_reach / prev_reach
                if ratio < threshold_ratio:  # 50% Abfall
                    events.append({
                        'step': history[i]['step'],
                        'from_reachability': prev_reach,
                        'to_reachability': curr_reach,
                        'ratio': float(ratio)
                    })

        if events:
            collapse_events[word] = events

    return collapse_events


def compute_plasticity_score(reachability_history):
    """
    Berechnet "Plastizitaet" pro Wort: Wie lange bleibt es beweglich?

    Returns:
        plasticity: Dict {word: plasticity_dict}
    """
    plasticity = {}

    for word, history in reachability_history.items():
        if len(history) < 2:
            plasticity[word] = {
                'mean_reachability': 0.0,
                'trend': 0.0,
                'final_reachability': 0.0,
                'max_reachability': 0.0,
                'plasticity_score': 0.0
            }
            continue

        # Reachability-Werte
        reaches = [h['reachability'] for h in history]

        # Plastizitaet = Durchschnittliche Reachability
        mean_reach = np.mean(reaches)

        # Trend: Steigt oder sinkt Reachability?
        steps = np.arange(len(reaches))
        if len(reaches) > 2:
            slope, _ = np.polyfit(steps, reaches, 1)
        else:
            slope = 0

        plasticity[word] = {
            'mean_reachability': float(mean_reach),
            'trend': float(slope),
            'final_reachability': float(reaches[-1]) if reaches else 0,
            'max_reachability': float(max(reaches)) if reaches else 0,
            'plasticity_score': float(mean_reach * (1 + slope * 10))  # Kombiniert
        }

    return plasticity


def correlate_with_frequency(plasticity, vocab_data):
    """
    Korreliert Plastizitaet mit Wort-Frequenz.

    Returns:
        correlation: Pearson-Korrelation
        data_points: Liste von (freq, plasticity) Tupeln
    """
    word_counts = vocab_data.get('word_counts', {})

    freqs = []
    plasticities = []
    data_points = []

    for word, pdata in plasticity.items():
        freq = word_counts.get(word, 1)
        plas = pdata['mean_reachability']

        freqs.append(freq)
        plasticities.append(plas)
        data_points.append({
            'word': word,
            'frequency': freq,
            'plasticity': plas
        })

    if len(freqs) > 2:
        correlation = float(np.corrcoef(freqs, plasticities)[0, 1])
    else:
        correlation = 0.0

    return correlation, data_points


def visualize_results(reachability_history, collapse_events, plasticity,
                      correlation, data_points, vocab_data, output_dir):
    """Erstellt Visualisierungen."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    word_counts = vocab_data.get('word_counts', {})

    # 1. Reachability ueber Zeit fuer einige Woerter
    ax = axes[0, 0]
    sample_words = list(reachability_history.keys())[:5]
    for word in sample_words:
        history = reachability_history[word]
        steps = [h['step'] for h in history]
        reaches = [h['reachability'] for h in history]
        ax.plot(steps, reaches, label=word, alpha=0.7)
    ax.set_xlabel('Trainingsschritt')
    ax.set_ylabel('Reachability')
    ax.set_title('Reachability Evolution (Sample Woerter)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Plasticity Ranking
    ax = axes[0, 1]
    if plasticity:
        sorted_words = sorted(plasticity.keys(),
                              key=lambda w: plasticity[w]['plasticity_score'],
                              reverse=True)
        top_words = sorted_words[:20]
        scores = [plasticity[w]['plasticity_score'] for w in top_words]

        ax.barh(range(len(top_words)), scores, alpha=0.7)
        ax.set_yticks(range(len(top_words)))
        ax.set_yticklabels(top_words, fontsize=8)
        ax.set_xlabel('Plasticity Score')
        ax.set_title('Top plastische Woerter')
        ax.invert_yaxis()

    # 3. Frequenz vs Plasticity
    ax = axes[0, 2]
    if data_points:
        freqs = [d['frequency'] for d in data_points]
        plas = [d['plasticity'] for d in data_points]
        ax.scatter(freqs, plas, alpha=0.7)
        for d in data_points[:10]:
            ax.annotate(d['word'], (d['frequency'], d['plasticity']), fontsize=7)
        ax.set_xlabel('Wort-Frequenz')
        ax.set_ylabel('Mean Reachability')
        ax.set_title(f'Frequenz vs Plasticity (r={correlation:.3f})')

    # 4. Collapse Events Timeline
    ax = axes[1, 0]
    all_collapses = []
    for word, events in collapse_events.items():
        for event in events:
            all_collapses.append({
                'word': word,
                'step': event['step'],
                'ratio': event['ratio']
            })

    if all_collapses:
        steps = [c['step'] for c in all_collapses]
        ratios = [c['ratio'] for c in all_collapses]
        ax.scatter(steps, ratios, alpha=0.5, s=20)
    ax.set_xlabel('Trainingsschritt')
    ax.set_ylabel('Collapse Ratio')
    ax.set_title(f'Collapse Events ({len(all_collapses)} Events)')
    ax.axhline(0.5, color='r', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # 5. Reachability Trend pro Wort
    ax = axes[1, 1]
    if plasticity:
        trends = [plasticity[w]['trend'] for w in plasticity.keys()]
        ax.hist(trends, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='r', linestyle='--')
    ax.set_xlabel('Reachability Trend')
    ax.set_ylabel('Anzahl Woerter')
    ax.set_title('Verteilung der Trends (< 0 = schrumpfend)')

    # 6. Zusammenfassung
    ax = axes[1, 2]
    ax.axis('off')

    n_collapsing = sum(1 for p in plasticity.values() if p['trend'] < 0)
    mean_plas = np.mean([p['mean_reachability'] for p in plasticity.values()])

    summary = f"""
    REACHABILITY HORIZON COLLAPSE (RHC)
    ===================================

    REACHABILITY-TRACKING:
    - Woerter analysiert: {len(reachability_history)}
    - Checkpoints: {len(list(reachability_history.values())[0]) if reachability_history else 0}

    PLASTICITY:
    - Mean Reachability: {mean_plas:.4f}
    - Schrumpfende Woerter: {n_collapsing} / {len(plasticity)}
    - Frequenz-Korrelation: {correlation:.3f}

    COLLAPSE EVENTS:
    - Woerter mit Collapse: {len(collapse_events)}
    - Total Events: {len(all_collapses)}

    TOP PLASTISCHE WOERTER:
    {chr(10).join(f"  {w}: {plasticity[w]['plasticity_score']:.4f}" for w in sorted(plasticity.keys(), key=lambda w: plasticity[w]['plasticity_score'], reverse=True)[:5])}

    INTERPRETATION:
    - Hohe Plasticity = Wort bleibt beweglich
    - Negativer Trend = Freiheit schrumpft
    - Collapse Events = Ploetzliche Einschraenkung
    - Positive Korrelation = Haeufige Woerter plastischer
    """
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'rhc_analysis.png')
    plt.savefig(output_path, dpi=150)
    print(f"  Gespeichert: {output_path}")
    plt.close()


def run_analysis(max_steps=None, checkpoint_interval=1000, horizon=100):
    """
    Fuehrt die vollstaendige RHC-Analyse durch.
    """
    output_dir = ensure_output_dir()

    print("=" * 70)
    print("REACHABILITY HORIZON COLLAPSE (RHC)")
    print("=" * 70)

    vocab_data = load_vocabulary()

    print(f"\n  Konfiguration:")
    print(f"    Checkpoint Interval: {checkpoint_interval}")
    print(f"    Horizon: {horizon}")

    with StepReader(STEPS_FILE) as reader:
        # Track Reachability
        print("\n[1/3] Tracke Reachability...")
        reachability_history = track_reachability_per_word(
            reader, vocab_data, checkpoint_interval, horizon, max_steps
        )
        print(f"  {len(reachability_history)} Woerter getrackt")

        # Detect Collapse Events
        print("\n[2/3] Detektiere Collapse Events...")
        collapse_events = detect_collapse_events(reachability_history)
        total_collapses = sum(len(e) for e in collapse_events.values())
        print(f"  {total_collapses} Collapse Events in {len(collapse_events)} Woertern")

        # Compute Plasticity
        print("\n[3/3] Berechne Plasticity...")
        plasticity = compute_plasticity_score(reachability_history)
        correlation, data_points = correlate_with_frequency(plasticity, vocab_data)
        print(f"  Frequenz-Korrelation: {correlation:.3f}")

        # Visualisierungen
        print("\n  Erstelle Visualisierungen...")
        visualize_results(reachability_history, collapse_events, plasticity,
                          correlation, data_points, vocab_data, output_dir)

    # Ergebnisse
    results = {
        'n_words_tracked': len(reachability_history),
        'n_words_with_collapse': len(collapse_events),
        'total_collapse_events': total_collapses,
        'frequency_correlation': float(correlation),
        'mean_plasticity': float(np.mean([p['mean_reachability'] for p in plasticity.values()])),
        'top_plastic_words': sorted(
            [(w, plasticity[w]['plasticity_score']) for w in plasticity.keys()],
            key=lambda x: x[1], reverse=True
        )[:10]
    }

    print("\n" + "=" * 70)
    print("RHC ANALYSE ABGESCHLOSSEN")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_analysis()
