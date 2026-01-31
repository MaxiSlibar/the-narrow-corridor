"""
Methode 11: Temporal Reversal Entropy (TRE)
==========================================

Konzept: Messe Irreversibilitaet durch Rueckwaerts-Analyse.
Wo kann man den "Zeitpfeil" erkennen? Wo ist Training irreversibel?

Ansatz:
- Fenster von N Change-Vektoren extrahieren
- Vorwaerts- und Rueckwaerts-Markov-Modelle bauen
- KL-Divergenz messen: D_KL(P_forward || P_backward)
- Hohe Divergenz = starke Irreversibilitaet

Frage beantwortet:
- Gibt es "Points of no return" im Training?
- Wann wird Training irreversibel?

Autor: Maximilian (MI-Forschung)
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

from .common import StepReader, STEPS_FILE, load_vocabulary
from .common.utils import ensure_output_dir


def discretize_changes(changes, n_bins=8):
    """
    Diskretisiert Change-Vektoren in Richtungskategorien.

    Verwendet Vorzeichen der Dimensionen als Binaercode (2^10 = 1024 moegliche).
    Fuer einfachere Analyse: Nur dominante Richtung (0-9) + Magnitude (low/high).

    Args:
        changes: Array (n, 10) von Change-Vektoren

    Returns:
        symbols: Liste von Symbolen (integers)
    """
    symbols = []

    # Median-Magnitude als Threshold
    magnitudes = np.linalg.norm(changes, axis=1)
    mag_threshold = np.median(magnitudes)

    for i, change in enumerate(changes):
        # Dominante Dimension (0-9)
        dominant_dim = np.argmax(np.abs(change))

        # Richtung (pos=0, neg=1)
        direction = 0 if change[dominant_dim] >= 0 else 1

        # Magnitude (low=0, high=1)
        mag_class = 0 if magnitudes[i] < mag_threshold else 1

        # Symbol = dim * 4 + direction * 2 + mag_class (0-39)
        symbol = dominant_dim * 4 + direction * 2 + mag_class
        symbols.append(symbol)

    return symbols


def build_transition_probs(symbols):
    """
    Baut Uebergangswahrscheinlichkeiten aus Symbol-Sequenz.

    Returns:
        probs: Dict {from_symbol: {to_symbol: probability}}
        counts: Dict {from_symbol: {to_symbol: count}}
    """
    counts = defaultdict(lambda: defaultdict(int))

    for i in range(len(symbols) - 1):
        from_sym = symbols[i]
        to_sym = symbols[i + 1]
        counts[from_sym][to_sym] += 1

    # Normalisiere
    probs = {}
    for from_sym, to_counts in counts.items():
        total = sum(to_counts.values())
        probs[from_sym] = {to_sym: c / total for to_sym, c in to_counts.items()}

    return probs, dict(counts)


def kl_divergence(p_probs, q_probs, epsilon=1e-10):
    """
    Berechnet symmetrische KL-Divergenz zwischen zwei Uebergangsmodellen.

    D_KL(P||Q) = sum_x P(x) * log(P(x) / Q(x))

    Returns:
        kl: Symmetrische KL-Divergenz
    """
    all_from_symbols = set(p_probs.keys()) | set(q_probs.keys())

    kl_pq = 0.0
    kl_qp = 0.0
    n_terms = 0

    for from_sym in all_from_symbols:
        p_dist = p_probs.get(from_sym, {})
        q_dist = q_probs.get(from_sym, {})

        all_to_symbols = set(p_dist.keys()) | set(q_dist.keys())

        for to_sym in all_to_symbols:
            p_val = p_dist.get(to_sym, epsilon)
            q_val = q_dist.get(to_sym, epsilon)

            # D_KL(P||Q)
            if p_val > epsilon:
                kl_pq += p_val * np.log(p_val / (q_val + epsilon))

            # D_KL(Q||P)
            if q_val > epsilon:
                kl_qp += q_val * np.log(q_val / (p_val + epsilon))

            n_terms += 1

    # Symmetrische Divergenz
    return (kl_pq + kl_qp) / 2


def compute_reversal_entropy(symbols, window_size=1000):
    """
    Berechnet Reversibilitaets-Entropie fuer ein Fenster.

    Returns:
        entropy: Symmetrische KL-Divergenz (vorwaerts vs rueckwaerts)
    """
    if len(symbols) < window_size:
        return 0.0

    # Vorwaerts
    forward_probs, _ = build_transition_probs(symbols)

    # Rueckwaerts
    reversed_symbols = list(reversed(symbols))
    backward_probs, _ = build_transition_probs(reversed_symbols)

    # KL-Divergenz
    return kl_divergence(forward_probs, backward_probs)


def find_irreversibility_profile(changes, window_size=1000, stride=500):
    """
    Berechnet Irreversibilitaets-Profil ueber gesamtes Training.

    Returns:
        profile: Liste von (step_idx, entropy) Tupeln
    """
    symbols = discretize_changes(changes)
    profile = []

    for start in range(0, len(symbols) - window_size, stride):
        window = symbols[start:start + window_size]
        entropy = compute_reversal_entropy(window, window_size)
        step_idx = start + window_size // 2
        profile.append((step_idx, entropy))

    return profile


def find_irreversibility_peaks(profile, threshold_percentile=90):
    """
    Findet Peaks der Irreversibilitaet (Points of no return).

    Returns:
        peaks: Liste von {step, entropy, rank}
    """
    if not profile:
        return []

    entropies = [e for _, e in profile]
    threshold = np.percentile(entropies, threshold_percentile)

    peaks = []
    for step, entropy in profile:
        if entropy > threshold:
            peaks.append({
                'step': step,
                'entropy': entropy,
                'percentile': (np.sum(np.array(entropies) <= entropy) / len(entropies)) * 100
            })

    # Sortiere nach Entropie
    peaks.sort(key=lambda x: x['entropy'], reverse=True)

    return peaks


def analyze_per_run(changes_per_run, window_size=500):
    """
    Analysiert Irreversibilitaet pro Run.

    Returns:
        run_profiles: Dict {run_id: profile}
        run_summary: Dict {run_id: summary_stats}
    """
    run_profiles = {}
    run_summary = {}

    for run_id, changes in changes_per_run.items():
        if len(changes) < window_size:
            continue

        profile = find_irreversibility_profile(changes, window_size, stride=window_size // 2)
        run_profiles[run_id] = profile

        if profile:
            entropies = [e for _, e in profile]
            run_summary[run_id] = {
                'mean_entropy': float(np.mean(entropies)),
                'max_entropy': float(np.max(entropies)),
                'std_entropy': float(np.std(entropies)),
                'n_windows': len(profile)
            }

    return run_profiles, run_summary


def analyze_epoch_transitions(symbols, steps_per_epoch=420):
    """
    Analysiert ob Epoch-Grenzen irreversibler sind.

    Returns:
        epoch_entropies: Entropie an jeder Epoch-Grenze
    """
    epoch_entropies = []
    window_size = steps_per_epoch // 2

    for epoch_end in range(steps_per_epoch, len(symbols), steps_per_epoch):
        start = max(0, epoch_end - window_size)
        end = min(len(symbols), epoch_end + window_size)
        window = symbols[start:end]

        if len(window) >= window_size:
            entropy = compute_reversal_entropy(window, len(window))
            epoch_entropies.append({
                'epoch': epoch_end // steps_per_epoch,
                'step': epoch_end,
                'entropy': entropy
            })

    return epoch_entropies


def visualize_results(profile, peaks, run_summary, epoch_entropies, output_dir):
    """Erstellt Visualisierungen."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Irreversibilitaets-Profil
    ax = axes[0, 0]
    if profile:
        steps, entropies = zip(*profile)
        ax.plot(steps, entropies, 'b-', alpha=0.7)
        ax.axhline(np.mean(entropies), color='r', linestyle='--', label='Mean')
        ax.axhline(np.percentile(entropies, 90), color='orange', linestyle='--', label='90th percentile')
    ax.set_xlabel('Trainingsschritt')
    ax.set_ylabel('Reversal Entropy (KL-Divergenz)')
    ax.set_title('Irreversibilitaets-Profil')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Peaks markiert
    ax = axes[0, 1]
    if profile:
        steps, entropies = zip(*profile)
        ax.plot(steps, entropies, 'b-', alpha=0.5)

        # Top 10 Peaks
        for peak in peaks[:10]:
            ax.axvline(peak['step'], color='r', alpha=0.5, linestyle='--')
            ax.annotate(f"{peak['entropy']:.3f}",
                        (peak['step'], peak['entropy']),
                        textcoords="offset points", xytext=(0, 10),
                        fontsize=7)

    ax.set_xlabel('Trainingsschritt')
    ax.set_ylabel('Reversal Entropy')
    ax.set_title(f'Points of No Return (Top {min(10, len(peaks))} Peaks)')
    ax.grid(True, alpha=0.3)

    # 3. Histogramm der Entropien
    ax = axes[0, 2]
    if profile:
        entropies = [e for _, e in profile]
        ax.hist(entropies, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(entropies), color='r', linestyle='--', label=f'Mean: {np.mean(entropies):.3f}')
    ax.set_xlabel('Reversal Entropy')
    ax.set_ylabel('Haeufigkeit')
    ax.set_title('Verteilung der Irreversibilitaet')
    ax.legend()

    # 4. Run-Vergleich
    ax = axes[1, 0]
    if run_summary:
        runs = sorted(run_summary.keys())
        mean_entropies = [run_summary[r]['mean_entropy'] for r in runs]
        ax.bar(runs, mean_entropies, alpha=0.7)
        ax.axhline(np.mean(mean_entropies), color='r', linestyle='--')
    ax.set_xlabel('Run ID')
    ax.set_ylabel('Mean Reversal Entropy')
    ax.set_title('Irreversibilitaet pro Run')

    # 5. Epoch-Analyse
    ax = axes[1, 1]
    if epoch_entropies:
        epochs = [e['epoch'] for e in epoch_entropies]
        entropies = [e['entropy'] for e in epoch_entropies]
        ax.scatter(epochs, entropies, alpha=0.5)
        ax.plot(epochs, entropies, 'b-', alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reversal Entropy an Epoch-Grenze')
    ax.set_title('Irreversibilitaet an Epoch-Grenzen')
    ax.grid(True, alpha=0.3)

    # 6. Zusammenfassung
    ax = axes[1, 2]
    ax.axis('off')

    if profile:
        entropies = [e for _, e in profile]
        mean_ent = np.mean(entropies)
        max_ent = np.max(entropies)
        std_ent = np.std(entropies)
    else:
        mean_ent = max_ent = std_ent = 0

    summary = f"""
    TEMPORAL REVERSAL ENTROPY (TRE)
    ===============================

    GESAMTANALYSE:
    - Fenster analysiert: {len(profile)}
    - Mean Entropy: {mean_ent:.4f}
    - Max Entropy: {max_ent:.4f}
    - Std Entropy: {std_ent:.4f}

    POINTS OF NO RETURN:
    - Peaks (>90. Perzentil): {len(peaks)}
    - Top Peak Step: {peaks[0]['step'] if peaks else 'N/A'}
    - Top Peak Entropy: {peaks[0]['entropy']:.4f} if peaks else 'N/A'

    RUN-VERGLEICH:
    - Runs analysiert: {len(run_summary)}
    - Mean Run-Entropy: {np.mean([r['mean_entropy'] for r in run_summary.values()]):.4f} if run_summary else 0

    INTERPRETATION:
    - Hohe Entropy = Training stark irreversibel
    - Peaks = Momente wo Rueckweg versperrt
    - Konstante Entropy = Stationaerer Prozess
    - Steigende Entropy = Zunehmende Irreversibilitaet
    """
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'tre_analysis.png')
    plt.savefig(output_path, dpi=150)
    print(f"  Gespeichert: {output_path}")
    plt.close()


def run_analysis(max_steps=None, window_size=1000, stride=500):
    """
    Fuehrt die vollstaendige TRE-Analyse durch.

    Args:
        max_steps: Maximale Schritte (None = alle)
        window_size: Groesse des Analyse-Fensters
        stride: Schrittweite zwischen Fenstern
    """
    output_dir = ensure_output_dir()

    print("=" * 70)
    print("TEMPORAL REVERSAL ENTROPY (TRE)")
    print("=" * 70)

    vocab_data = load_vocabulary()
    config = vocab_data['config']
    steps_per_epoch = config['num_pairs']
    steps_per_run = steps_per_epoch * config['epochs']

    print(f"\n  Konfiguration:")
    print(f"    Window Size: {window_size}")
    print(f"    Stride: {stride}")
    print(f"    Steps per Epoch: {steps_per_epoch}")

    with StepReader(STEPS_FILE) as reader:
        # Lade Changes
        print("\n[1/4] Lade Change-Vektoren...")
        changes = reader.get_all_changes(max_steps)
        print(f"  {len(changes):,} Changes geladen")

        # Globales Profil
        print("\n[2/4] Berechne Irreversibilitaets-Profil...")
        profile = find_irreversibility_profile(changes, window_size, stride)
        print(f"  {len(profile)} Fenster analysiert")

        # Peaks finden
        print("\n[3/4] Finde Points of No Return...")
        peaks = find_irreversibility_peaks(profile)
        print(f"  {len(peaks)} Peaks gefunden")

        # Per-Run Analyse
        print("\n[4/4] Analysiere pro Run...")
        # Teile Changes in Runs
        n_steps = len(changes)
        n_runs = n_steps // steps_per_run

        changes_per_run = {}
        for run_id in range(min(n_runs, 50)):
            start = run_id * steps_per_run
            end = start + steps_per_run
            if end <= len(changes):
                changes_per_run[run_id] = changes[start:end]

        run_profiles, run_summary = analyze_per_run(changes_per_run, window_size=500)
        print(f"  {len(run_summary)} Runs analysiert")

        # Epoch-Analyse
        symbols = discretize_changes(changes)
        epoch_entropies = analyze_epoch_transitions(symbols, steps_per_epoch)
        print(f"  {len(epoch_entropies)} Epoch-Grenzen analysiert")

        # Visualisierungen
        print("\n  Erstelle Visualisierungen...")
        visualize_results(profile, peaks, run_summary, epoch_entropies, output_dir)

    # Ergebnisse
    if profile:
        entropies = [e for _, e in profile]
        mean_entropy = float(np.mean(entropies))
        max_entropy = float(np.max(entropies))
    else:
        mean_entropy = max_entropy = 0.0

    results = {
        'n_windows': len(profile),
        'mean_entropy': mean_entropy,
        'max_entropy': max_entropy,
        'n_peaks': len(peaks),
        'top_peaks': peaks[:5],
        'n_runs_analyzed': len(run_summary),
        'run_summary': run_summary,
        'epoch_analysis': {
            'n_epochs': len(epoch_entropies),
            'mean_epoch_entropy': float(np.mean([e['entropy'] for e in epoch_entropies])) if epoch_entropies else 0
        }
    }

    print("\n" + "=" * 70)
    print("TRE ANALYSE ABGESCHLOSSEN")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_analysis()
