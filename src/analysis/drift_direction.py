"""
Methode 7: Drift Direction Census (DDC)
=======================================

Konzept: Quantisiere die Richtung jeder Embedding-Aenderung in diskrete Bins
(Oktanten in 3D PCA-Raum). Zaehle wie oft jede Richtung genutzt wird.

Frage beantwortet:
- #5 (Richtung von Drift)
- #8 (neue Auswertungsfragen)

Hypothese: Woerter entwickeln "bevorzugte Drift-Kanaele" - emergente
Koordinatensysteme im Embedding-Raum.

Autor: Maximilian (MI-Forschung)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import defaultdict
import os

from .common import StepReader, STEPS_FILE, load_vocabulary
from .common.utils import ensure_output_dir


# Wortklassen basierend auf NEUEM Trainingstext
# Aktualisiert fuer vocabulary.json mit 34 Woertern (MIN_WORD_COUNT >= 2)
WORD_CLASSES = {
    'artikel': ['der', 'die', 'das', 'den'],
    'nomen_tier': ['katze', 'hund', 'maus', 'vogel'],
    'nomen_person': ['mann', 'frau', 'kind', 'mutter'],
    'verb_bewegung': ['jagt', 'faehrt', 'fliegt', 'schwimmt', 'steht'],
    'verb_sonstig': ['trinkt', 'spielt', 'arbeitet', 'schlaeft', 'heilt', 'scheint', 'waechst', 'lehrt', 'liest'],
    'adjektiv': ['schnell', 'langsam', 'hoch', 'tief', 'leise', 'laut'],
    'sonstige': ['ist', 'viel']
}


def get_word_class(word):
    """
    Gibt die Wortklasse fuer ein Wort zurueck.

    Args:
        word: Das Wort als String

    Returns:
        Wortklassen-Name oder 'unbekannt'
    """
    word_lower = word.lower()
    for cls, words in WORD_CLASSES.items():
        if word_lower in words:
            return cls
    return 'unbekannt'


def analyze_entropy_by_word_class(word_histograms, word_entropy, idx_to_word):
    """
    Analysiert ob Wortklassen-Gruppen niedrigere Entropie haben.

    Hypothese: Woerter gleicher Klasse teilen Drift-Muster, d.h.
    wenn wir ihre Histogramme kombinieren, sollte die kombinierte
    Entropie niedriger sein als die mittlere individuelle Entropie.

    Args:
        word_histograms: Dict mapping word_idx -> histogram
        word_entropy: Dict mapping word_idx -> entropy
        idx_to_word: Dict mapping idx -> word

    Returns:
        class_analysis: Dict mit Analyse pro Wortklasse
    """
    class_entropies = defaultdict(list)
    class_histograms = defaultdict(lambda: np.zeros(8))

    for word_idx, entropy in word_entropy.items():
        word = idx_to_word.get(str(word_idx), f"word_{word_idx}")
        word_class = get_word_class(word)
        class_entropies[word_class].append(entropy)
        class_histograms[word_class] += word_histograms[word_idx]

    results = {}
    for cls in class_entropies:
        individual_entropies = class_entropies[cls]
        mean_individual = np.mean(individual_entropies)

        # Kombinierte Entropie der Klasse
        combined_hist = class_histograms[cls]
        combined_entropy = compute_direction_entropy(combined_hist) if combined_hist.sum() > 0 else 0

        # Entropie-Reduktion: Wenn positiv, hat die Klasse gemeinsame Drift-Muster
        entropy_reduction = mean_individual - combined_entropy

        results[cls] = {
            'num_words': len(individual_entropies),
            'mean_individual_entropy': float(mean_individual),
            'combined_entropy': float(combined_entropy),
            'entropy_reduction': float(entropy_reduction),
            'words': [idx_to_word.get(str(idx), '?') for idx in word_entropy.keys()
                      if get_word_class(idx_to_word.get(str(idx), '')) == cls]
        }

    return results


def visualize_word_class_analysis(class_analysis, output_dir):
    """Visualisiert die Wortklassen-Entropie-Analyse."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Sortiere Klassen nach mittlerer Entropie
    sorted_classes = sorted(class_analysis.items(),
                            key=lambda x: x[1]['mean_individual_entropy'])

    classes = [c for c, _ in sorted_classes]
    mean_entropies = [d['mean_individual_entropy'] for _, d in sorted_classes]
    combined_entropies = [d['combined_entropy'] for _, d in sorted_classes]
    num_words = [d['num_words'] for _, d in sorted_classes]

    # 1. Entropie-Vergleich
    ax = axes[0]
    x = np.arange(len(classes))
    width = 0.35

    bars1 = ax.bar(x - width/2, mean_entropies, width,
                   label='Mittlere individuelle Entropie', color='steelblue', alpha=0.7)
    bars2 = ax.bar(x + width/2, combined_entropies, width,
                   label='Kombinierte Klassen-Entropie', color='coral', alpha=0.7)

    ax.set_xlabel('Wortklasse')
    ax.set_ylabel('Entropie (bits)')
    ax.set_title('Drift-Entropie nach Wortklasse')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.axhline(3.0, color='red', linestyle='--', alpha=0.5, label='Max = 3 bits')
    ax.grid(True, alpha=0.3)

    # Anzahl Woerter als Annotation
    for i, n in enumerate(num_words):
        ax.annotate(f'n={n}', (x[i], max(mean_entropies[i], combined_entropies[i]) + 0.1),
                    ha='center', fontsize=8)

    # 2. Entropie-Reduktion
    ax = axes[1]
    reductions = [d['entropy_reduction'] for _, d in sorted_classes]
    colors = ['green' if r > 0 else 'red' for r in reductions]

    ax.barh(classes, reductions, color=colors, alpha=0.7)
    ax.set_xlabel('Entropie-Reduktion (bits)')
    ax.set_ylabel('Wortklasse')
    ax.set_title('Entropie-Reduktion durch Klassenzugehoerigkeit')
    ax.axvline(0, color='black', linewidth=1)
    ax.grid(True, alpha=0.3)

    # Interpretation
    ax.text(0.02, 0.02,
            'Positiv = Klasse hat gemeinsame Drift-Muster\nNegativ = Klasse heterogen',
            transform=ax.transAxes, fontsize=8, verticalalignment='bottom')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'ddc_word_class_analysis.png')
    plt.savefig(output_path, dpi=150)
    print(f"  Gespeichert: {output_path}")
    plt.close()


def fit_direction_pca(all_changes, n_components=3):
    """
    Fittet PCA auf alle Change-Vektoren.

    Args:
        all_changes: Array (N, D)
        n_components: Anzahl PCA-Komponenten

    Returns:
        pca: Gefittetes PCA-Modell
    """
    pca = PCA(n_components=n_components)
    pca.fit(all_changes)
    return pca


def quantize_direction(change, pca_model):
    """
    Quantisiert Aenderungsrichtung in Oktant.

    Projiziert auf 3D PCA-Raum und bestimmt Oktant basierend auf Vorzeichen.

    Args:
        change: Aenderungsvektor
        pca_model: PCA-Modell

    Returns:
        octant_id: Integer 0-7
    """
    projected = pca_model.transform([change])[0]

    # Quantisiere: positive = 1, negativ/null = 0
    octant_bits = (np.sign(projected) > 0).astype(int)

    # Konvertiere zu Oktant-ID (0-7)
    octant_id = octant_bits @ np.array([1, 2, 4])

    return int(octant_id)


def build_direction_histogram(changes, pca_model):
    """
    Baut Histogramm der Drift-Richtungen.

    Args:
        changes: Array (N, D)
        pca_model: PCA-Modell

    Returns:
        histogram: Array (8,) mit Zaehlungen pro Oktant
    """
    octants = []
    for change in changes:
        octant = quantize_direction(change, pca_model)
        octants.append(octant)

    histogram = np.bincount(octants, minlength=8)
    return histogram


def compute_direction_entropy(histogram):
    """
    Berechnet Entropie der Richtungsverteilung.

    Hohe Entropie = gleichmaessig verteilte Richtungen
    Niedrige Entropie = bevorzugte Richtungen

    Returns:
        entropy: Entropie in bits
    """
    probs = histogram / histogram.sum()
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    return entropy


def analyze_per_word_drift(step_reader, vocabulary, pca_model, max_steps=None):
    """
    Analysiert Drift-Richtungen pro Wort.

    Returns:
        word_histograms: Dict mapping word -> histogram
        word_entropy: Dict mapping word -> entropy
    """
    word_changes = defaultdict(list)

    n = min(step_reader.num_steps, max_steps) if max_steps else step_reader.num_steps

    print("  Sammle Changes pro Wort...")
    for i in range(n):
        step = step_reader.read_step(i)
        word_idx = step['target_idx']
        word_changes[word_idx].append(step['change'])

        if (i + 1) % 100000 == 0:
            print(f"    {i+1:,} / {n:,}")

    word_histograms = {}
    word_entropy = {}

    for word_idx, changes in word_changes.items():
        changes = np.array(changes)
        hist = build_direction_histogram(changes, pca_model)
        ent = compute_direction_entropy(hist)

        word_histograms[word_idx] = hist
        word_entropy[word_idx] = ent

    return word_histograms, word_entropy


def find_shared_drift_channels(word_histograms, threshold=0.2):
    """
    Findet Oktanten die von vielen Woertern bevorzugt werden.

    Args:
        word_histograms: Dict mapping word -> histogram
        threshold: Mindestanteil fuer "bevorzugt"

    Returns:
        shared_channels: Dict mapping octant -> list of words
    """
    shared_channels = defaultdict(list)

    for word_idx, hist in word_histograms.items():
        # Normalisiere
        probs = hist / hist.sum()

        # Finde bevorzugte Oktanten
        for octant in range(8):
            if probs[octant] > threshold:
                shared_channels[octant].append(word_idx)

    return dict(shared_channels)


def analyze_drift_evolution(step_reader, pca_model, window_size=10000, step=5000):
    """
    Analysiert wie sich Drift-Praeferenzen ueber das Training entwickeln.

    Returns:
        evolution: Array (n_windows, 8)
        window_centers: Zeitpunkte
    """
    n = step_reader.num_steps

    evolution = []
    window_centers = []

    for start in range(0, n - window_size, step):
        changes = []
        for i in range(start, start + window_size):
            step_data = step_reader.read_step(i)
            changes.append(step_data['change'])

        hist = build_direction_histogram(np.array(changes), pca_model)
        evolution.append(hist / hist.sum())
        window_centers.append((start + start + window_size) // 2)

    return np.array(evolution), np.array(window_centers)


def octant_to_name(octant_id):
    """Gibt einen beschreibenden Namen fuer den Oktanten."""
    signs = [(octant_id >> i) & 1 for i in range(3)]
    components = ['PC1', 'PC2', 'PC3']
    parts = []
    for i, s in enumerate(signs):
        parts.append(f"+{components[i]}" if s else f"-{components[i]}")
    return "/".join(parts)


def visualize_results(pca_model, global_hist, word_histograms, word_entropy,
                      shared_channels, evolution, window_centers,
                      idx_to_word, output_dir):
    """Erstellt Visualisierungen der DDC-Analyse."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Globales Oktanten-Histogramm
    ax = axes[0, 0]
    octant_names = [octant_to_name(i) for i in range(8)]
    ax.bar(range(8), global_hist, color='steelblue', alpha=0.7)
    ax.set_xticks(range(8))
    ax.set_xticklabels(octant_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Haeufigkeit')
    ax.set_title('Globale Drift-Richtungsverteilung')
    ax.grid(True, alpha=0.3)

    # 2. PCA Explained Variance
    ax = axes[0, 1]
    explained = pca_model.explained_variance_ratio_
    ax.bar(range(len(explained)), explained, color='green', alpha=0.7)
    ax.set_xlabel('PCA Komponente')
    ax.set_ylabel('Erklaerte Varianz')
    ax.set_title(f'PCA: {sum(explained)*100:.1f}% Varianz erklaert')
    ax.grid(True, alpha=0.3)

    # 3. Entropie pro Wort
    ax = axes[0, 2]
    words = [idx_to_word[str(idx)] for idx in sorted(word_entropy.keys())]
    entropies = [word_entropy[idx] for idx in sorted(word_entropy.keys())]

    sorted_idx = np.argsort(entropies)
    ax.barh(range(len(words)), [entropies[i] for i in sorted_idx], color='purple', alpha=0.7)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels([words[i] for i in sorted_idx], fontsize=8)
    ax.set_xlabel('Richtungs-Entropie (bits)')
    ax.set_title('Woerter nach Drift-Diversitaet')
    ax.axvline(3.0, color='red', linestyle='--', label='Max Entropie = 3 bits')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Shared Drift Channels
    ax = axes[1, 0]
    channel_counts = [len(shared_channels.get(o, [])) for o in range(8)]
    colors = plt.cm.Reds(np.array(channel_counts) / max(channel_counts) if max(channel_counts) > 0 else np.zeros(8))
    ax.bar(range(8), channel_counts, color=colors)
    ax.set_xticks(range(8))
    ax.set_xticklabels(octant_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Anzahl Woerter')
    ax.set_title('Gemeinsame Drift-Kanaele (>20% Praeferenz)')
    ax.grid(True, alpha=0.3)

    # 5. Drift Evolution
    ax = axes[1, 1]
    for octant in range(8):
        ax.plot(window_centers, evolution[:, octant], alpha=0.7,
                label=octant_to_name(octant))
    ax.set_xlabel('Trainingsschritt')
    ax.set_ylabel('Anteil')
    ax.set_title('Evolution der Drift-Richtungen')
    ax.legend(fontsize=6, loc='upper right')
    ax.grid(True, alpha=0.3)

    # 6. Zusammenfassung
    ax = axes[1, 2]
    ax.axis('off')

    global_entropy = compute_direction_entropy(global_hist)
    dominant_octant = np.argmax(global_hist)

    summary = f"""
    DRIFT DIRECTION CENSUS - ZUSAMMENFASSUNG

    PCA ANALYSE:
    - Erklaerte Varianz: {sum(explained)*100:.1f}%
    - Komponente 1: {explained[0]*100:.1f}%
    - Komponente 2: {explained[1]*100:.1f}%
    - Komponente 3: {explained[2]*100:.1f}%

    GLOBALE DRIFT:
    - Dominanter Oktant: {octant_to_name(dominant_octant)}
    - Anteil: {global_hist[dominant_octant]/global_hist.sum()*100:.1f}%
    - Richtungs-Entropie: {global_entropy:.2f} bits (max 3.0)

    WORT-ANALYSE:
    - Mittlere Entropie: {np.mean(list(word_entropy.values())):.2f} bits
    - Min Entropie: {min(word_entropy.values()):.2f} bits
    - Max Entropie: {max(word_entropy.values()):.2f} bits

    EMERGENTE KANAELE:
    - Oktanten mit >3 Woertern: {sum(1 for c in channel_counts if c > 3)}
    - Meist geteilter Kanal: {octant_to_name(np.argmax(channel_counts))}
      ({max(channel_counts)} Woerter)
    """
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'ddc_analysis.png')
    plt.savefig(output_path, dpi=150)
    print(f"  Gespeichert: {output_path}")
    plt.close()


def run_analysis(max_steps=None):
    """
    Fuehrt die vollstaendige DDC-Analyse durch.
    """
    output_dir = ensure_output_dir()
    print("=" * 70)
    print("DRIFT DIRECTION CENSUS (DDC) ANALYSE")
    print("=" * 70)

    vocab_data = load_vocabulary()
    vocabulary = vocab_data['vocabulary']
    idx_to_word = vocab_data['idx_to_word']

    with StepReader(STEPS_FILE) as reader:
        # 1. Alle Changes laden
        print("\n  Lade Change-Vektoren...")
        n = min(reader.num_steps, max_steps) if max_steps else reader.num_steps
        all_changes = reader.get_all_changes(n)
        print(f"  Geladen: {len(all_changes):,} Vektoren")

        # 2. PCA fitten
        print("\n  Fitte PCA...")
        pca_model = fit_direction_pca(all_changes, n_components=3)
        print(f"  Erklaerte Varianz: {pca_model.explained_variance_ratio_}")

        # 3. Globales Histogramm
        print("\n  Baue globales Richtungs-Histogramm...")
        global_hist = build_direction_histogram(all_changes, pca_model)
        global_entropy = compute_direction_entropy(global_hist)
        print(f"  Globale Entropie: {global_entropy:.2f} bits")

        # 4. Pro-Wort Analyse
        print("\n  Analysiere Drift pro Wort...")
        word_histograms, word_entropy = analyze_per_word_drift(
            reader, vocabulary, pca_model, max_steps
        )

        # 5. Shared Channels
        shared_channels = find_shared_drift_channels(word_histograms)
        print(f"\n  Gefundene gemeinsame Kanaele:")
        for octant, words in shared_channels.items():
            word_names = [idx_to_word[str(w)] for w in words[:5]]
            print(f"    {octant_to_name(octant)}: {word_names}...")

        # 6. Evolution
        print("\n  Analysiere Drift-Evolution...")
        evolution, window_centers = analyze_drift_evolution(reader, pca_model)

        # 7. Wortklassen-Analyse
        print("\n  Analysiere Entropie nach Wortklassen...")
        class_analysis = analyze_entropy_by_word_class(word_histograms, word_entropy, idx_to_word)
        print(f"  Entropie nach Klasse:")
        for cls, data in sorted(class_analysis.items(), key=lambda x: x[1]['mean_individual_entropy']):
            print(f"    {cls}: {data['mean_individual_entropy']:.2f} bits ({data['num_words']} Woerter)")

        # 8. Visualisierungen
        print("\n  Erstelle Visualisierungen...")
        visualize_results(pca_model, global_hist, word_histograms, word_entropy,
                          shared_channels, evolution, window_centers,
                          idx_to_word, output_dir)
        visualize_word_class_analysis(class_analysis, output_dir)

    # Ergebnisse
    results = {
        'pca_variance_explained': pca_model.explained_variance_ratio_.tolist(),
        'global_histogram': global_hist.tolist(),
        'global_entropy': float(global_entropy),
        'word_entropy': {idx_to_word[str(k)]: v for k, v in word_entropy.items()},
        'shared_channels': {octant_to_name(k): [idx_to_word[str(w)] for w in v]
                           for k, v in shared_channels.items()},
        'word_class_analysis': class_analysis
    }

    print("\n" + "=" * 70)
    print("DDC ANALYSE ABGESCHLOSSEN")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_analysis()
