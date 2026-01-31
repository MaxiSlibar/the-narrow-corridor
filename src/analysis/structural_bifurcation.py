"""
Methode 14: Structural Bifurcation Detection (SBD)
=================================================

Konzept: Finde singulaere Momente die globale Struktur permanent aendern.
Bifurkationspunkte = "Weggabelungen" im Training.

Ansatz:
- Pseudo-Jacobian aus Fenster von Changes schaetzen (Kovarianz)
- Eigenwerte der Kovarianz-Matrix tracken
- Bifurkation = plotzliche Aenderung der Eigenwert-Struktur

Frage beantwortet:
- Wann "committed" das Training zu bestimmten Pfaden?
- Warum divergieren verschiedene Runs?

Autor: Maximilian (MI-Forschung)
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

from .common import StepReader, STEPS_FILE, load_vocabulary
from .common.utils import ensure_output_dir


def compute_local_covariance(changes):
    """
    Berechnet Kovarianz-Matrix der Changes als Proxy fuer lokale Dynamik.

    Returns:
        cov: (10, 10) Kovarianz-Matrix
        eigenvalues: Sortierte Eigenwerte
        eigenvectors: Eigenvektoren
    """
    if len(changes) < 2:
        return np.eye(10), np.ones(10), np.eye(10)

    cov = np.cov(changes.T)

    # Eigenwerte/Eigenvektoren
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sortiere absteigend
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return cov, eigenvalues, eigenvectors


def compute_eigenvalue_metrics(eigenvalues, epsilon=1e-10):
    """
    Berechnet verschiedene Metriken aus Eigenwerten.

    Returns:
        metrics: Dict mit verschiedenen Kennzahlen
    """
    eigenvalues = np.maximum(eigenvalues, epsilon)

    return {
        'ratio': float(eigenvalues[0] / eigenvalues[-1]),
        'entropy': float(-np.sum((eigenvalues / eigenvalues.sum()) * np.log(eigenvalues / eigenvalues.sum() + epsilon))),
        'effective_dim': float(np.exp(-np.sum((eigenvalues / eigenvalues.sum()) * np.log(eigenvalues / eigenvalues.sum() + epsilon)))),
        'condition_number': float(eigenvalues[0] / eigenvalues[-1]),
        'trace': float(np.sum(eigenvalues)),
        'dominant_explained': float(eigenvalues[0] / np.sum(eigenvalues))
    }


def detect_bifurcations(step_reader, window_size=1000, stride=500, max_steps=None):
    """
    Detektiert Bifurkationspunkte durch Eigenwert-Analyse.

    Returns:
        all_metrics: Liste von {step, eigenvalues, metrics}
        bifurcations: Liste von {step, type, strength}
    """
    all_metrics = []
    bifurcations = []

    n = min(step_reader.num_steps, max_steps) if max_steps else step_reader.num_steps

    prev_metrics = None
    prev_eigenvalues = None

    print(f"  Analysiere {(n - window_size) // stride} Fenster...")

    for start in range(0, n - window_size, stride):
        # Lade Changes fuer dieses Fenster
        changes = []
        for i in range(start, start + window_size):
            step = step_reader.read_step(i)
            changes.append(step['change'])
        changes = np.array(changes)

        # Berechne Kovarianz und Eigenwerte
        cov, eigenvalues, eigenvectors = compute_local_covariance(changes)
        metrics = compute_eigenvalue_metrics(eigenvalues)

        step_idx = start + window_size // 2
        metrics['step'] = step_idx
        metrics['eigenvalues'] = eigenvalues.tolist()

        all_metrics.append(metrics)

        # Detektiere Bifurkationen
        if prev_metrics is not None:
            # Ratio-Aenderung
            ratio_change = abs(metrics['ratio'] - prev_metrics['ratio'])

            # Entropy-Aenderung
            entropy_change = abs(metrics['entropy'] - prev_metrics['entropy'])

            # Eigenwert-Kreuzungen (Vorzeichen-Wechsel der Differenzen)
            if prev_eigenvalues is not None:
                eigenvalue_changes = eigenvalues - prev_eigenvalues
                sign_changes = np.sum(np.diff(np.sign(eigenvalue_changes)) != 0)
            else:
                sign_changes = 0

            # Kombinierter Bifurkations-Score
            bifurcation_score = ratio_change + entropy_change * 10

            if bifurcation_score > 0.5:  # Threshold
                bif_type = classify_bifurcation(prev_metrics, metrics, eigenvalues, prev_eigenvalues)
                bifurcations.append({
                    'step': step_idx,
                    'type': bif_type,
                    'score': float(bifurcation_score),
                    'ratio_change': float(ratio_change),
                    'entropy_change': float(entropy_change),
                    'sign_changes': int(sign_changes)
                })

        prev_metrics = metrics
        prev_eigenvalues = eigenvalues.copy()

        if len(all_metrics) % 100 == 0:
            print(f"    {len(all_metrics)} Fenster analysiert")

    # Sortiere Bifurkationen nach Score
    bifurcations.sort(key=lambda x: x['score'], reverse=True)

    return all_metrics, bifurcations


def classify_bifurcation(prev_metrics, curr_metrics, curr_eigenvalues, prev_eigenvalues):
    """
    Klassifiziert den Typ der Bifurkation.

    Returns:
        type: String Beschreibung
    """
    ratio_increased = curr_metrics['ratio'] > prev_metrics['ratio']
    entropy_increased = curr_metrics['entropy'] > prev_metrics['entropy']

    if ratio_increased and not entropy_increased:
        return "anisotropy_increase"  # Bevorzugte Richtung wird staerker
    elif not ratio_increased and entropy_increased:
        return "isotropy_increase"  # Wird gleichmaessiger
    elif ratio_increased and entropy_increased:
        return "complexity_increase"  # Wird komplexer
    else:
        return "simplification"  # Wird einfacher


def compare_runs(step_reader, config, max_runs=10, window_size=500):
    """
    Vergleicht Bifurkationen zwischen verschiedenen Runs.

    Returns:
        run_bifurcations: Dict {run_id: [bifurcations]}
        common_steps: Schritte wo mehrere Runs Bifurkationen haben
    """
    steps_per_run = config['num_pairs'] * config['epochs']
    run_bifurcations = {}

    for run_id in range(min(max_runs, 50)):
        start = run_id * steps_per_run
        end = start + steps_per_run

        if end > step_reader.num_steps:
            break

        # Analysiere diesen Run
        changes = []
        for i in range(start, min(end, start + 10000)):  # Sample
            step = step_reader.read_step(i)
            changes.append(step['change'])

        if len(changes) < window_size:
            continue

        changes = np.array(changes)

        # Finde Bifurkationen in diesem Run
        run_bifs = []
        prev_metrics = None

        for w_start in range(0, len(changes) - window_size, window_size // 2):
            window = changes[w_start:w_start + window_size]
            _, eigenvalues, _ = compute_local_covariance(window)
            metrics = compute_eigenvalue_metrics(eigenvalues)

            if prev_metrics is not None:
                score = abs(metrics['ratio'] - prev_metrics['ratio'])
                if score > 0.5:
                    # Relative Position im Run (0-1)
                    rel_pos = w_start / len(changes)
                    run_bifs.append({
                        'relative_position': rel_pos,
                        'score': float(score)
                    })

            prev_metrics = metrics

        run_bifurcations[run_id] = run_bifs

    # Finde gemeinsame Bifurkations-Positionen
    all_positions = []
    for run_id, bifs in run_bifurcations.items():
        for bif in bifs:
            all_positions.append(bif['relative_position'])

    # Histogram der Positionen
    if all_positions:
        hist, bin_edges = np.histogram(all_positions, bins=20)
        common_positions = []
        threshold = np.percentile(hist[hist > 0], 70) if np.any(hist > 0) else 0
        for i, count in enumerate(hist):
            if count > threshold:
                common_positions.append({
                    'position_range': (float(bin_edges[i]), float(bin_edges[i + 1])),
                    'n_runs': int(count)
                })
    else:
        common_positions = []

    return run_bifurcations, common_positions


def visualize_results(all_metrics, bifurcations, run_bifurcations, common_positions, output_dir):
    """Erstellt Visualisierungen."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Eigenvalue Ratio ueber Zeit
    ax = axes[0, 0]
    if all_metrics:
        steps = [m['step'] for m in all_metrics]
        ratios = [min(m['ratio'], 100) for m in all_metrics]
        ax.plot(steps, ratios, 'b-', alpha=0.7)
        ax.set_yscale('log')

        # Markiere Bifurkationen
        for bif in bifurcations[:20]:
            ax.axvline(bif['step'], color='r', alpha=0.3, linestyle='--')

    ax.set_xlabel('Trainingsschritt')
    ax.set_ylabel('Eigenvalue Ratio (log)')
    ax.set_title('Strukturelle Anisotropie (rot = Bifurkationen)')
    ax.grid(True, alpha=0.3)

    # 2. Entropy ueber Zeit
    ax = axes[0, 1]
    if all_metrics:
        steps = [m['step'] for m in all_metrics]
        entropy = [m['entropy'] for m in all_metrics]
        ax.plot(steps, entropy, 'g-', alpha=0.7)

        for bif in bifurcations[:20]:
            ax.axvline(bif['step'], color='r', alpha=0.3, linestyle='--')

    ax.set_xlabel('Trainingsschritt')
    ax.set_ylabel('Eigenwert-Entropie')
    ax.set_title('Strukturelle Komplexitaet')
    ax.grid(True, alpha=0.3)

    # 3. Bifurkations-Scores
    ax = axes[0, 2]
    if bifurcations:
        top_bifs = bifurcations[:30]
        steps = [b['step'] for b in top_bifs]
        scores = [b['score'] for b in top_bifs]
        colors = {'anisotropy_increase': 'red', 'isotropy_increase': 'blue',
                  'complexity_increase': 'purple', 'simplification': 'green'}
        c = [colors.get(b['type'], 'gray') for b in top_bifs]

        ax.scatter(steps, scores, c=c, alpha=0.7, s=50)

    ax.set_xlabel('Trainingsschritt')
    ax.set_ylabel('Bifurkations-Score')
    ax.set_title('Top Bifurkationen (Farbe = Typ)')
    ax.grid(True, alpha=0.3)

    # 4. Bifurkations-Typen Verteilung
    ax = axes[1, 0]
    if bifurcations:
        types = [b['type'] for b in bifurcations]
        unique_types, counts = np.unique(types, return_counts=True)
        ax.bar(unique_types, counts, alpha=0.7)
        ax.set_xticklabels(unique_types, rotation=45, ha='right')
    ax.set_ylabel('Anzahl')
    ax.set_title('Bifurkations-Typen')

    # 5. Run-Vergleich
    ax = axes[1, 1]
    if run_bifurcations:
        for run_id, bifs in list(run_bifurcations.items())[:5]:
            if bifs:
                positions = [b['relative_position'] for b in bifs]
                scores = [b['score'] for b in bifs]
                ax.scatter(positions, [run_id] * len(positions), s=[s * 50 for s in scores], alpha=0.5)

    ax.set_xlabel('Relative Position im Run')
    ax.set_ylabel('Run ID')
    ax.set_title('Bifurkationen pro Run')

    # 6. Zusammenfassung
    ax = axes[1, 2]
    ax.axis('off')

    if bifurcations:
        types_count = defaultdict(int)
        for b in bifurcations:
            types_count[b['type']] += 1

    summary = f"""
    STRUCTURAL BIFURCATION DETECTION (SBD)
    =====================================

    GESAMTANALYSE:
    - Fenster analysiert: {len(all_metrics)}
    - Bifurkationen gefunden: {len(bifurcations)}

    BIFURKATIONS-TYPEN:
    {chr(10).join(f'  {t}: {c}' for t, c in types_count.items()) if bifurcations else '  Keine'}

    TOP 5 BIFURKATIONEN:
    {chr(10).join(f'  Step {b["step"]}: {b["type"]} (Score: {b["score"]:.2f})' for b in bifurcations[:5]) if bifurcations else '  Keine'}

    RUN-VERGLEICH:
    - Runs analysiert: {len(run_bifurcations)}
    - Gemeinsame Positionen: {len(common_positions)}
    {chr(10).join(f'  {cp["position_range"]}: {cp["n_runs"]} Runs' for cp in common_positions[:3]) if common_positions else '  Keine'}

    INTERPRETATION:
    - anisotropy_increase = Bevorzugte Richtung staerker
    - isotropy_increase = Wird gleichmaessiger
    - complexity_increase = Mehr Freiheitsgrade
    - simplification = Weniger Freiheitsgrade
    """
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'sbd_analysis.png')
    plt.savefig(output_path, dpi=150)
    print(f"  Gespeichert: {output_path}")
    plt.close()


def run_analysis(max_steps=None, window_size=1000, stride=500):
    """
    Fuehrt die vollstaendige SBD-Analyse durch.
    """
    output_dir = ensure_output_dir()

    print("=" * 70)
    print("STRUCTURAL BIFURCATION DETECTION (SBD)")
    print("=" * 70)

    vocab_data = load_vocabulary()
    config = vocab_data['config']

    print(f"\n  Konfiguration:")
    print(f"    Window Size: {window_size}")
    print(f"    Stride: {stride}")

    with StepReader(STEPS_FILE) as reader:
        # Detektiere Bifurkationen
        print("\n[1/2] Detektiere Bifurkationen...")
        all_metrics, bifurcations = detect_bifurcations(
            reader, window_size, stride, max_steps
        )
        print(f"  {len(all_metrics)} Fenster analysiert")
        print(f"  {len(bifurcations)} Bifurkationen gefunden")

        # Vergleiche Runs
        print("\n[2/2] Vergleiche Runs...")
        run_bifurcations, common_positions = compare_runs(reader, config, max_runs=10)
        print(f"  {len(run_bifurcations)} Runs analysiert")
        print(f"  {len(common_positions)} gemeinsame Positionen")

        # Visualisierungen
        print("\n  Erstelle Visualisierungen...")
        visualize_results(all_metrics, bifurcations, run_bifurcations, common_positions, output_dir)

    # Ergebnisse
    type_counts = defaultdict(int)
    for b in bifurcations:
        type_counts[b['type']] += 1

    results = {
        'n_windows': len(all_metrics),
        'n_bifurcations': len(bifurcations),
        'bifurcation_types': dict(type_counts),
        'top_bifurcations': bifurcations[:10],
        'run_comparison': {
            'n_runs': len(run_bifurcations),
            'common_positions': common_positions
        },
        'mean_ratio': float(np.mean([m['ratio'] for m in all_metrics if m['ratio'] < 1000])) if all_metrics else 0,
        'mean_entropy': float(np.mean([m['entropy'] for m in all_metrics])) if all_metrics else 0
    }

    print("\n" + "=" * 70)
    print("SBD ANALYSE ABGESCHLOSSEN")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_analysis()
