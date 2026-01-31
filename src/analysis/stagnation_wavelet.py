"""
Methode 4: Stagnation Wavelet Transform (SWT)
=============================================

Konzept: Erstelle binaeres "Stagnationssignal" (1 wenn Aenderung klein, 0 sonst),
wende Wavelet-Analyse an um charakteristische Stagnationsfrequenzen zu finden.

Frage beantwortet:
- #5 (Haeufigkeit von Stagnation)
- #4 (Strukturen bei langen Laeufen)

Hypothese: Lernen hat charakteristische Rhythmen - ein "Herzschlag" des
Gradient Descent.

Autor: Maximilian (MI-Forschung)
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt
import os

from .common import StepReader, STEPS_FILE, load_vocabulary
from .common.utils import ensure_output_dir


def create_stagnation_signal(change_magnitudes, threshold=None):
    """
    Erstellt binaeres Stagnationssignal.

    Args:
        change_magnitudes: Array der Aenderungsbetraege
        threshold: Schwelle fuer Stagnation (None = Median)

    Returns:
        signal: Binaeres Array (1 = stagniert, 0 = aktiv)
    """
    if threshold is None:
        threshold = np.median(change_magnitudes)

    signal = (change_magnitudes < threshold).astype(float)
    return signal, threshold


def wavelet_transform(signal, scales=None, wavelet='morl'):
    """
    Fuehrt Continuous Wavelet Transform durch.

    Args:
        signal: Eingangssignal
        scales: Skalen fuer CWT (None = automatisch)
        wavelet: Wavelet-Typ

    Returns:
        coeffs: Wavelet-Koeffizienten
        freqs: Korrespondierende Frequenzen
    """
    if scales is None:
        scales = np.arange(1, min(128, len(signal) // 10))

    coeffs, freqs = pywt.cwt(signal, scales, wavelet)
    return coeffs, freqs, scales


def find_dominant_frequencies(coeffs, scales, top_n=10):
    """
    Findet die dominanten Frequenzen im Signal.

    Returns:
        dominant: Liste von (scale, power) Tupeln
    """
    # Mittlere Power pro Skala
    power_per_scale = np.mean(np.abs(coeffs)**2, axis=1)

    # Top N Skalen
    top_indices = np.argsort(power_per_scale)[::-1][:top_n]

    return [(int(scales[i]), float(power_per_scale[i])) for i in top_indices]


def find_stagnation_beats(signal, min_period=10, max_period=500):
    """
    Findet periodische Muster im Stagnationssignal.

    Returns:
        periods: Gefundene Perioden mit Staerke
    """
    # Autokorrelation
    n = len(signal)
    autocorr = np.correlate(signal - np.mean(signal), signal - np.mean(signal), mode='full')
    autocorr = autocorr[n-1:]  # Nur positive Lags
    autocorr = autocorr / autocorr[0]  # Normalisieren

    # Finde Peaks in der Autokorrelation
    periods = []
    for i in range(min_period, min(max_period, len(autocorr) - 1)):
        if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
            if autocorr[i] > 0.1:  # Signifikanter Peak
                periods.append((i, float(autocorr[i])))

    return sorted(periods, key=lambda x: x[1], reverse=True)[:20]


def analyze_stagnation_phases(signal, min_run_length=50):
    """
    Analysiert laengere Stagnations- und Aktivitaets-Phasen.

    Returns:
        stagnation_runs: Liste von (start, length) fuer Stagnationsphasen
        activity_runs: Liste von (start, length) fuer Aktivitaetsphasen
    """
    stagnation_runs = []
    activity_runs = []

    current_start = 0
    current_type = signal[0]

    for i in range(1, len(signal)):
        if signal[i] != current_type:
            length = i - current_start
            if length >= min_run_length:
                if current_type == 1:
                    stagnation_runs.append((current_start, length))
                else:
                    activity_runs.append((current_start, length))
            current_start = i
            current_type = signal[i]

    # Letzter Run
    length = len(signal) - current_start
    if length >= min_run_length:
        if current_type == 1:
            stagnation_runs.append((current_start, length))
        else:
            activity_runs.append((current_start, length))

    return stagnation_runs, activity_runs


def compare_runs_spectra(reader, num_runs=50, steps_per_run=18000):
    """
    Vergleicht Wavelet-Spektren ueber verschiedene Runs.

    Returns:
        spectra: Dict mapping run_id -> spectrum
        similarity_matrix: Paarweise Aehnlichkeit der Spektren
    """
    spectra = {}
    scales = np.arange(1, 128)

    for run_id in range(num_runs):  # Alle Runs fuer vollstaendige Analyse
        if (run_id + 1) % 10 == 0:
            print(f"    Processing run {run_id + 1}/{num_runs}...")
        # Hole Daten fuer diesen Run
        magnitudes = []
        start_idx = run_id * steps_per_run

        for i in range(steps_per_run):
            step = reader.read_step(start_idx + i)
            magnitudes.append(step['change_magnitude'])

        signal, _ = create_stagnation_signal(np.array(magnitudes))
        coeffs, _, _ = wavelet_transform(signal, scales)

        # Power-Spektrum
        power = np.mean(np.abs(coeffs)**2, axis=1)
        spectra[run_id] = power

    # Similarity Matrix
    run_ids = list(spectra.keys())
    n = len(run_ids)
    similarity = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # Korrelation der Power-Spektren
            similarity[i, j] = np.corrcoef(spectra[run_ids[i]], spectra[run_ids[j]])[0, 1]

    return spectra, similarity


def visualize_results(signal, coeffs, scales, dominant, beats, stag_runs, act_runs,
                      spectra, similarity, output_dir):
    """Erstellt Visualisierungen der SWT-Analyse."""

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    # 1. Stagnationssignal
    ax = axes[0, 0]
    ax.plot(signal[:10000], 'b-', alpha=0.5, linewidth=0.3)
    ax.fill_between(range(min(10000, len(signal))), signal[:10000], alpha=0.3)
    ax.set_xlabel('Schritt')
    ax.set_ylabel('Stagnation (1) / Aktiv (0)')
    ax.set_title('Stagnationssignal (erste 10k Schritte)')
    ax.grid(True, alpha=0.3)

    # 2. Scalogram (Wavelet-Koeffizienten)
    ax = axes[0, 1]
    im = ax.imshow(np.abs(coeffs[:, :10000]), aspect='auto', cmap='viridis',
                   extent=[0, 10000, scales[-1], scales[0]])
    ax.set_xlabel('Schritt')
    ax.set_ylabel('Skala')
    ax.set_title('Wavelet Scalogram (erste 10k Schritte)')
    plt.colorbar(im, ax=ax, label='|Koeffizient|')

    # 3. Dominante Frequenzen
    ax = axes[1, 0]
    dom_scales = [d[0] for d in dominant]
    dom_powers = [d[1] for d in dominant]
    ax.barh(range(len(dominant)), dom_powers, color='purple', alpha=0.7)
    ax.set_yticks(range(len(dominant)))
    ax.set_yticklabels([f'Skala {s}' for s in dom_scales])
    ax.set_xlabel('Power')
    ax.set_title('Top 10 Dominante Skalen')
    ax.grid(True, alpha=0.3)

    # 4. Stagnations-Beats (Autokorrelation)
    ax = axes[1, 1]
    if beats:
        periods = [b[0] for b in beats]
        strengths = [b[1] for b in beats]
        ax.bar(periods, strengths, color='green', alpha=0.7, width=5)
        ax.set_xlabel('Periode (Schritte)')
        ax.set_ylabel('Autokorrelation')
        ax.set_title('Gefundene Perioditaeten')
    ax.grid(True, alpha=0.3)

    # 5. Phasen-Laengenverteilung
    ax = axes[2, 0]
    if stag_runs:
        stag_lengths = [r[1] for r in stag_runs]
        ax.hist(stag_lengths, bins=30, alpha=0.5, label='Stagnation', color='red')
    if act_runs:
        act_lengths = [r[1] for r in act_runs]
        ax.hist(act_lengths, bins=30, alpha=0.5, label='Aktivitaet', color='blue')
    ax.set_xlabel('Phasen-Laenge')
    ax.set_ylabel('Haeufigkeit')
    ax.set_title('Verteilung der Phasen-Laengen')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Spektren-Vergleich ueber Runs
    ax = axes[2, 1]
    if spectra:
        for run_id, spectrum in spectra.items():
            ax.plot(scales, spectrum, alpha=0.7, label=f'Run {run_id}')
        ax.set_xlabel('Skala')
        ax.set_ylabel('Power')
        ax.set_title('Spektren-Vergleich zwischen Runs')
        ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'swt_analysis.png')
    plt.savefig(output_path, dpi=150)
    print(f"  Gespeichert: {output_path}")
    plt.close()

    # Zusaetzlich: Similarity Matrix
    if similarity is not None and len(similarity) > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(similarity, cmap='RdYlGn', vmin=-1, vmax=1)
        ax.set_xlabel('Run')
        ax.set_ylabel('Run')
        ax.set_title('Spektrale Aehnlichkeit zwischen Runs')
        plt.colorbar(im, ax=ax)
        output_path = os.path.join(output_dir, 'swt_similarity.png')
        plt.savefig(output_path, dpi=150)
        print(f"  Gespeichert: {output_path}")
        plt.close()


def run_analysis(max_steps=None):
    """
    Fuehrt die vollstaendige SWT-Analyse durch.
    """
    output_dir = ensure_output_dir()
    print("=" * 70)
    print("STAGNATION WAVELET TRANSFORM (SWT) ANALYSE")
    print("=" * 70)

    # Vokabular laden
    vocab_data = load_vocabulary()
    config = vocab_data['config']
    steps_per_run = config['num_pairs'] * config['epochs']

    with StepReader(STEPS_FILE) as reader:
        # 1. Change Magnitudes laden
        print("\n  Lade Change Magnitudes...")
        n = min(reader.num_steps, max_steps) if max_steps else reader.num_steps
        magnitudes = reader.get_all_change_magnitudes(n)
        print(f"  Geladen: {len(magnitudes):,} Werte")

        # 2. Stagnationssignal erstellen
        print("\n  Erstelle Stagnationssignal...")
        signal, threshold = create_stagnation_signal(magnitudes)
        stagnation_rate = np.mean(signal)
        print(f"  Threshold: {threshold:.6f}")
        print(f"  Stagnationsrate: {stagnation_rate:.1%}")

        # 3. Wavelet Transform
        print("\n  Fuehre Wavelet Transform durch...")
        scales = np.arange(1, 128)
        coeffs, freqs, _ = wavelet_transform(signal, scales)
        print(f"  Koeffizienten-Shape: {coeffs.shape}")

        # 4. Dominante Frequenzen
        dominant = find_dominant_frequencies(coeffs, scales)
        print(f"\n  Top 5 dominante Skalen:")
        for scale, power in dominant[:5]:
            print(f"    Skala {scale}: Power {power:.4f}")

        # 5. Stagnations-Beats
        print("\n  Suche Stagnations-Beats...")
        beats = find_stagnation_beats(signal)
        if beats:
            print(f"  Gefundene Perioden:")
            for period, strength in beats[:5]:
                print(f"    Periode {period}: Staerke {strength:.3f}")

        # 6. Phasen-Analyse
        print("\n  Analysiere Stagnations-Phasen...")
        stag_runs, act_runs = analyze_stagnation_phases(signal)
        print(f"  Lange Stagnationsphasen: {len(stag_runs)}")
        print(f"  Lange Aktivitaetsphasen: {len(act_runs)}")

        # 7. Spektren-Vergleich
        print("\n  Vergleiche Spektren zwischen Runs...")
        spectra, similarity = compare_runs_spectra(reader, num_runs=50, steps_per_run=steps_per_run)
        if len(similarity) > 1:
            mean_sim = np.mean(similarity[np.triu_indices_from(similarity, k=1)])
            print(f"  Mittlere spektrale Aehnlichkeit: {mean_sim:.3f}")

        # 8. Visualisierungen
        print("\n  Erstelle Visualisierungen...")
        visualize_results(signal, coeffs, scales, dominant, beats, stag_runs, act_runs,
                          spectra, similarity, output_dir)

    # Ergebnisse zusammenfassen
    results = {
        'stagnation_threshold': float(threshold),
        'stagnation_rate': float(stagnation_rate),
        'dominant_scales': dominant,
        'beats': beats,
        'num_stagnation_phases': len(stag_runs),
        'num_activity_phases': len(act_runs),
        'mean_stagnation_length': float(np.mean([r[1] for r in stag_runs])) if stag_runs else 0,
        'mean_activity_length': float(np.mean([r[1] for r in act_runs])) if act_runs else 0
    }

    print("\n" + "=" * 70)
    print("SWT ANALYSE ABGESCHLOSSEN")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_analysis()
