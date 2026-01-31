"""
Methode 2: Curvature Flow Decomposition (CFD)
=============================================

Konzept: Behandle Embeddings als Punkte auf einer Mannigfaltigkeit.
Miss lokale Kruemmung bei jedem Schritt. Zerlege Trajektorie in
geodaetische Segmente.

Frage beantwortet:
- #7 (Speicher als Bewegungsgeometrie)

Hypothese: "Erinnerung" korreliert mit geometrischer Geradlinigkeit -
Woerter die eine Richtung fanden und beibehielten haben staerkere
Repraesentationen.

Autor: Maximilian (MI-Forschung)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from .common import EmbeddingReader, EMBEDDINGS_FILE, load_vocabulary
from .common.utils import ensure_output_dir


def compute_curvature(trajectory):
    """
    Berechnet diskrete Frenet-Kruemmung fuer hochdimensionale Raeume.

    Kruemmung = |a_senkrecht| / |v|^2

    Wobei a_senkrecht die Komponente der Beschleunigung senkrecht
    zur Geschwindigkeit ist.

    Args:
        trajectory: numpy Array (N, D) - N Zeitpunkte, D Dimensionen

    Returns:
        curvature: numpy Array (N-2,) - Kruemmung pro Schritt
    """
    # Geschwindigkeit (erste Ableitung)
    velocity = np.diff(trajectory, axis=0)  # Shape: (N-1, D)

    # Beschleunigung (zweite Ableitung)
    acceleration = np.diff(velocity, axis=0)  # Shape: (N-2, D)

    # Geschwindigkeit und Beschleunigung ausrichten
    v = velocity[1:]  # Shape: (N-2, D)
    a = acceleration   # Shape: (N-2, D)

    # Geschwindigkeit (Betrag)
    speed = np.linalg.norm(v, axis=1)  # Shape: (N-2,)

    # Einheitsvektor der Geschwindigkeit
    v_norm = v / (speed[:, None] + 1e-10)

    # Parallele Komponente der Beschleunigung: (a . v_norm) * v_norm
    a_parallel_magnitude = np.sum(a * v_norm, axis=1, keepdims=True)
    a_parallel = a_parallel_magnitude * v_norm

    # Senkrechte Komponente: a - a_parallel
    a_perp = a - a_parallel

    # Kruemmung = |a_perp| / |v|^2
    curvature = np.linalg.norm(a_perp, axis=1) / (speed**2 + 1e-10)

    return curvature


def compute_torsion(trajectory):
    """
    Berechnet Torsion (Verdrehung aus der Oskulationsebene).

    Torsion misst, wie stark die Trajektorie aus ihrer lokalen
    Kruemmungsebene herausdreht.

    Args:
        trajectory: numpy Array (N, D)

    Returns:
        torsion: numpy Array (N-3,)
    """
    v = np.diff(trajectory, axis=0)
    a = np.diff(v, axis=0)
    j = np.diff(a, axis=0)  # Jerk (dritte Ableitung)

    # Fuer Torsion benoetigen wir das Spatprodukt (v x a) . j
    # In hohen Dimensionen ist das komplizierter - wir approximieren

    # Torsion ~ |j_senkrecht_zu_(v,a)| / |a|^2
    v_aligned = v[2:]
    a_aligned = a[1:]

    # Projektion von j auf die Ebene aufgespannt durch v und a
    v_norm = v_aligned / (np.linalg.norm(v_aligned, axis=1, keepdims=True) + 1e-10)
    a_perp_v = a_aligned - np.sum(a_aligned * v_norm, axis=1, keepdims=True) * v_norm
    a_perp_norm = a_perp_v / (np.linalg.norm(a_perp_v, axis=1, keepdims=True) + 1e-10)

    j_parallel_v = np.sum(j * v_norm, axis=1, keepdims=True) * v_norm
    j_parallel_a = np.sum(j * a_perp_norm, axis=1, keepdims=True) * a_perp_norm
    j_perp = j - j_parallel_v - j_parallel_a

    torsion = np.linalg.norm(j_perp, axis=1) / (np.linalg.norm(a_aligned, axis=1)**2 + 1e-10)

    return torsion


def segment_by_curvature(curvature, threshold_low=0.01, threshold_high=0.1):
    """
    Segmentiert Trajektorie nach Kruemmungsregime.

    Args:
        curvature: Kruemmungswerte
        threshold_low: Unter diesem Wert = geodaetisch
        threshold_high: Ueber diesem Wert = Wendepunkt

    Returns:
        segments: Liste von (start, end, type) Tupeln
    """
    segments = []
    current_start = 0
    current_type = None

    for i, k in enumerate(curvature):
        if k < threshold_low:
            new_type = 'geodesic'
        elif k > threshold_high:
            new_type = 'turning'
        else:
            new_type = 'transition'

        if current_type is None:
            current_type = new_type
        elif new_type != current_type:
            segments.append((current_start, i, current_type))
            current_start = i
            current_type = new_type

    # Letztes Segment
    segments.append((current_start, len(curvature), current_type))

    return segments


def compute_geodesic_ratio(segments):
    """
    Berechnet Anteil geodaetischer Bewegung.

    Returns:
        ratio: Anteil der Zeit in geodaetischer Bewegung (0-1)
    """
    total_length = sum(end - start for start, end, _ in segments)
    geodesic_length = sum(
        end - start for start, end, stype in segments
        if stype == 'geodesic'
    )
    return geodesic_length / total_length if total_length > 0 else 0


def compute_total_distance(trajectory):
    """Berechnet Gesamtdistanz der Trajektorie."""
    return np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))


def compute_displacement(trajectory):
    """Berechnet direkte Verschiebung (Start zu Ende)."""
    return np.linalg.norm(trajectory[-1] - trajectory[0])


def compute_efficiency(trajectory):
    """
    Berechnet Bewegungs-Effizienz = Verschiebung / Gesamtdistanz.

    1.0 = perfekt gerade
    < 1.0 = Umwege, Maeander
    """
    total = compute_total_distance(trajectory)
    displacement = compute_displacement(trajectory)
    return displacement / total if total > 0 else 0


def analyze_frequency_effect(all_results, vocabulary, word_counts):
    """
    Untersucht ob Worthaeufigkeit geodaetisches Verhalten beeinflusst.

    Hypothese: Haeufige Woerter nehmen mehr "Umwege" (niedrigerer
    geodaetischer Anteil) weil sie von vielen Kontexten beeinflusst werden.

    WICHTIG: word_counts enthaelt ALLE Korpus-Woerter, aber nur
    Woerter im vocabulary (MIN_WORD_COUNT >= 2) sind relevant.

    Args:
        all_results: Liste der Analyse-Ergebnisse pro Wort
        vocabulary: word -> idx Mapping
        word_counts: word -> Haeufigkeit im Korpus

    Returns:
        dict mit Frequenz-Analyse Ergebnissen
    """
    # Frequenz zu jedem Ergebnis hinzufuegen
    for r in all_results:
        r['frequency'] = word_counts.get(r['word'], 0)

    freqs = [r['frequency'] for r in all_results]
    geos = [r['geodesic_ratio'] for r in all_results]

    # Korrelation berechnen
    freq_geo_corr = np.corrcoef(freqs, geos)[0, 1] if len(freqs) > 1 else 0.0

    # Gruppen: haeufig (>=5x) vs selten (<5x)
    frequent = [r for r in all_results if r['frequency'] >= 5]
    rare = [r for r in all_results if r['frequency'] < 5]

    return {
        'freq_geodesic_correlation': float(freq_geo_corr),
        'frequent_words': [(r['word'], r['frequency'], r['geodesic_ratio'])
                           for r in frequent],
        'rare_words': [(r['word'], r['frequency'], r['geodesic_ratio'])
                       for r in rare],
        'frequent_mean_geo': float(np.mean([r['geodesic_ratio'] for r in frequent])) if frequent else 0,
        'rare_mean_geo': float(np.mean([r['geodesic_ratio'] for r in rare])) if rare else 0,
        'interpretation': "Negative Korrelation = haeufige Woerter nehmen mehr Umwege"
    }


def analyze_word_geometry(emb_reader, word_idx, word_name, run_length,
                          threshold_low=0.01, threshold_high=0.1):
    """
    Analysiert die Bewegungsgeometrie eines einzelnen Wortes.
    """
    # Trajektorie lesen (nur erster Run)
    trajectory = emb_reader.read_word_trajectory(word_idx, 0, run_length)

    # Kruemmung berechnen
    curvature = compute_curvature(trajectory)

    # Torsion berechnen
    torsion = compute_torsion(trajectory)

    # Segmentieren
    segments = segment_by_curvature(curvature, threshold_low, threshold_high)

    # Metriken
    geodesic_ratio = compute_geodesic_ratio(segments)
    efficiency = compute_efficiency(trajectory)
    total_distance = compute_total_distance(trajectory)
    displacement = compute_displacement(trajectory)

    return {
        'word': word_name,
        'trajectory': trajectory,
        'curvature': curvature,
        'torsion': torsion,
        'segments': segments,
        'geodesic_ratio': geodesic_ratio,
        'efficiency': efficiency,
        'total_distance': total_distance,
        'displacement': displacement,
        'mean_curvature': np.mean(curvature),
        'max_curvature': np.max(curvature),
        'mean_torsion': np.mean(torsion),
        'max_torsion': np.max(torsion)
    }


def visualize_word_geometry(result, output_path):
    """Visualisiert die Geometrie-Analyse eines Wortes."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    word = result['word']

    # 1. Kruemmung ueber Zeit
    ax = axes[0, 0]
    ax.plot(result['curvature'], 'b-', alpha=0.7, linewidth=0.5)
    ax.axhline(0.01, color='green', linestyle='--', label='Geodaetisch-Schwelle')
    ax.axhline(0.1, color='red', linestyle='--', label='Wendepunkt-Schwelle')
    ax.set_xlabel('Schritt')
    ax.set_ylabel('Kruemmung')
    ax.set_title(f"'{word}' - Kruemmung ueber Zeit")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 2. Kruemmungsverteilung
    ax = axes[0, 1]
    ax.hist(result['curvature'], bins=100, color='blue', alpha=0.7, log=True)
    ax.axvline(0.01, color='green', linestyle='--')
    ax.axvline(0.1, color='red', linestyle='--')
    ax.set_xlabel('Kruemmung')
    ax.set_ylabel('Haeufigkeit (log)')
    ax.set_title(f"'{word}' - Kruemmungsverteilung")
    ax.grid(True, alpha=0.3)

    # 3. Segmente visualisieren
    ax = axes[1, 0]
    colors = {'geodesic': 'green', 'transition': 'yellow', 'turning': 'red'}
    for start, end, stype in result['segments']:
        ax.axvspan(start, end, alpha=0.3, color=colors[stype], label=stype)

    # Legende ohne Duplikate
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    ax.plot(result['curvature'], 'k-', alpha=0.5, linewidth=0.3)
    ax.set_xlabel('Schritt')
    ax.set_ylabel('Kruemmung')
    ax.set_title(f"'{word}' - Segmentierung (Geodaetisch: {result['geodesic_ratio']:.1%})")
    ax.grid(True, alpha=0.3)

    # 4. Metriken-Zusammenfassung
    ax = axes[1, 1]
    ax.axis('off')
    metrics_text = f"""
    Wort: '{word}'

    BEWEGUNGSMETRIKEN:
    - Geodaetischer Anteil: {result['geodesic_ratio']:.1%}
    - Effizienz (Verschiebung/Distanz): {result['efficiency']:.3f}
    - Gesamtdistanz: {result['total_distance']:.2f}
    - Direkte Verschiebung: {result['displacement']:.2f}

    KRUEMMUNGSSTATISTIK:
    - Mittlere Kruemmung: {result['mean_curvature']:.4f}
    - Max Kruemmung: {result['max_curvature']:.4f}

    TORSIONSSTATISTIK:
    - Mittlere Torsion: {result['mean_torsion']:.6f}
    - Max Torsion: {result['max_torsion']:.6f}

    SEGMENTE:
    - Geodaetisch: {sum(1 for s in result['segments'] if s[2]=='geodesic')}
    - Uebergang: {sum(1 for s in result['segments'] if s[2]=='transition')}
    - Wendepunkt: {sum(1 for s in result['segments'] if s[2]=='turning')}
    """
    ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def visualize_comparison(all_results, output_dir):
    """Vergleicht Geometrie-Metriken aller Woerter."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    words = [r['word'] for r in all_results]
    geodesic_ratios = [r['geodesic_ratio'] for r in all_results]
    efficiencies = [r['efficiency'] for r in all_results]
    mean_curvatures = [r['mean_curvature'] for r in all_results]
    displacements = [r['displacement'] for r in all_results]

    # Sortiere nach geodaetischem Anteil
    sorted_idx = np.argsort(geodesic_ratios)[::-1]

    # 1. Geodaetischer Anteil
    ax = axes[0, 0]
    ax.barh(range(len(words)), [geodesic_ratios[i] for i in sorted_idx], color='green', alpha=0.7)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels([words[i] for i in sorted_idx])
    ax.set_xlabel('Geodaetischer Anteil')
    ax.set_title('Woerter nach geodaetischer Bewegung')
    ax.grid(True, alpha=0.3)

    # 2. Effizienz vs Geodaetisch
    ax = axes[0, 1]
    ax.scatter(geodesic_ratios, efficiencies, s=100, alpha=0.7)
    for i, word in enumerate(words):
        ax.annotate(word, (geodesic_ratios[i], efficiencies[i]), fontsize=8)
    ax.set_xlabel('Geodaetischer Anteil')
    ax.set_ylabel('Effizienz')
    ax.set_title('Geodaetischer Anteil vs Effizienz')
    ax.grid(True, alpha=0.3)

    # 3. Mittlere Kruemmung vs Verschiebung
    ax = axes[1, 0]
    ax.scatter(mean_curvatures, displacements, s=100, alpha=0.7, c=geodesic_ratios, cmap='RdYlGn')
    for i, word in enumerate(words):
        ax.annotate(word, (mean_curvatures[i], displacements[i]), fontsize=8)
    ax.set_xlabel('Mittlere Kruemmung')
    ax.set_ylabel('Gesamtverschiebung')
    ax.set_title('Kruemmung vs Verschiebung (Farbe = Geodaetisch)')
    ax.grid(True, alpha=0.3)

    # 4. Zusammenfassungsstatistik
    ax = axes[1, 1]
    ax.axis('off')

    # Korrelation berechnen
    corr_geo_eff = np.corrcoef(geodesic_ratios, efficiencies)[0, 1]
    corr_curv_disp = np.corrcoef(mean_curvatures, displacements)[0, 1]

    summary = f"""
    ZUSAMMENFASSUNG CURVATURE FLOW DECOMPOSITION

    Analysierte Woerter: {len(words)}

    KORRELATIONEN:
    - Geodaetisch <-> Effizienz: {corr_geo_eff:.3f}
    - Kruemmung <-> Verschiebung: {corr_curv_disp:.3f}

    TOP 5 GEODAETISCHE WOERTER:
    {chr(10).join(f'  {words[i]}: {geodesic_ratios[i]:.1%}' for i in sorted_idx[:5])}

    BOTTOM 5 (WENDUNGSREICH):
    {chr(10).join(f'  {words[i]}: {geodesic_ratios[i]:.1%}' for i in sorted_idx[-5:])}

    INTERPRETATION:
    - Hoher geodaetischer Anteil = stabile Bewegungsrichtung
    - Hohe Effizienz = direkter Weg zum Ziel
    - Hypothese: "Geodaetische" Woerter = staerkere Repraesentationen
    """
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'cfd_comparison.png')
    plt.savefig(output_path, dpi=150)
    print(f"  Gespeichert: {output_path}")
    plt.close()


def run_analysis(words_to_analyze=None, max_steps=None):
    """
    Fuehrt die vollstaendige CFD-Analyse durch.

    Args:
        words_to_analyze: Liste von Woertern (None = alle)
        max_steps: Max Schritte pro Wort (None = ein Run)

    Returns:
        dict mit allen Ergebnissen
    """
    output_dir = ensure_output_dir()
    print("=" * 70)
    print("CURVATURE FLOW DECOMPOSITION (CFD) ANALYSE")
    print("=" * 70)

    # Vokabular laden
    vocab_data = load_vocabulary()
    config = vocab_data['config']
    vocabulary = vocab_data['vocabulary']
    idx_to_word = vocab_data['idx_to_word']

    run_length = config['num_pairs'] * config['epochs'] + 1
    if max_steps:
        run_length = min(run_length, max_steps)

    print(f"\nKonfiguration:")
    print(f"  Schritte pro Run: {run_length:,}")
    print(f"  Dimensionen: {config['embedding_dim']}")

    # Woerter auswaehlen
    if words_to_analyze is None:
        words_to_analyze = list(vocabulary.keys())

    print(f"\nAnalysiere {len(words_to_analyze)} Woerter...")

    all_results = []

    with EmbeddingReader(EMBEDDINGS_FILE) as reader:
        for word in words_to_analyze:
            if word not in vocabulary:
                print(f"  Wort '{word}' nicht im Vokabular, ueberspringe.")
                continue

            word_idx = vocabulary[word]
            print(f"\n  Analysiere '{word}' (idx={word_idx})...")

            result = analyze_word_geometry(
                reader, word_idx, word, run_length,
                threshold_low=0.01, threshold_high=0.1
            )
            all_results.append(result)

            # Einzelne Visualisierung
            output_path = os.path.join(output_dir, f'cfd_{word}.png')
            visualize_word_geometry(result, output_path)
            print(f"    Geodaetisch: {result['geodesic_ratio']:.1%}, "
                  f"Effizienz: {result['efficiency']:.3f}")

    # Vergleichende Visualisierung
    print("\n  Erstelle Vergleichs-Visualisierung...")
    visualize_comparison(all_results, output_dir)

    # Frequenz-Effekt analysieren
    word_counts = vocab_data.get('word_counts', {})
    freq_analysis = analyze_frequency_effect(all_results, vocabulary, word_counts)
    print(f"\n  Frequenz-Analyse:")
    print(f"    Korrelation Frequenz <-> Geodaetisch: {freq_analysis['freq_geodesic_correlation']:.3f}")
    print(f"    Haeufige Woerter (>=5x): {freq_analysis['frequent_mean_geo']:.1%} geodaetisch")
    print(f"    Seltene Woerter (<5x): {freq_analysis['rare_mean_geo']:.1%} geodaetisch")

    # Ergebnisse zusammenfassen
    results = {
        'num_words': len(all_results),
        'frequency_analysis': freq_analysis,
        'word_results': {r['word']: {
            'geodesic_ratio': r['geodesic_ratio'],
            'efficiency': r['efficiency'],
            'mean_curvature': r['mean_curvature'],
            'max_curvature': r['max_curvature'],
            'displacement': r['displacement'],
            'total_distance': r['total_distance'],
            'frequency': r.get('frequency', 0)
        } for r in all_results},
        'ranking_geodesic': sorted(
            [(r['word'], r['geodesic_ratio']) for r in all_results],
            key=lambda x: x[1], reverse=True
        )
    }

    print("\n" + "=" * 70)
    print("CFD ANALYSE ABGESCHLOSSEN")
    print("=" * 70)

    return results


if __name__ == "__main__":
    # Teste mit allen Woertern
    results = run_analysis()
