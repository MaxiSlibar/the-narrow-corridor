"""
Methode 9: Absence Cartography (AC)
===================================

Konzept: Kartiere was NICHT passiert ist. Fuer jedes moegliche (target, context)
Paar: Wie oft haette es trainiert werden koennen vs. wie oft wurde es trainiert?
Analysiere die Geometrie von "Trainingsloechern".

Frage beantwortet:
- #5 (Ausbleiben erwarteter Uebergaenge)

Hypothese: Der "Negativraum" des Trainings enthuellt ebenso viel wie das
Training selbst.

Autor: Maximilian (MI-Forschung)
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

from .common import StepReader, STEPS_FILE, load_vocabulary
from .common.utils import ensure_output_dir


# TRAINING_CORPUS - identisch zu experiment_detailed_logging.py
TRAINING_CORPUS = """
Die Katze jagt die Maus. Der Hund jagt den Ball. Der Fuchs jagt das Huhn.
Die Aerztin heilt den Mann. Der Arzt heilt die Frau. Die Mutter heilt das Kind.
Das Auto faehrt schnell. Der Bus faehrt langsam. Das Fahrrad faehrt leise.
Der Vogel fliegt hoch. Das Flugzeug fliegt hoeher. Die Biene fliegt tief.
Die Katze trinkt Milch. Der Hund trinkt Wasser. Das Kind trinkt Saft.
Der Baecker backt Brot. Die Koechin kocht Suppe. Der Koch bratet Fleisch.
Die Sonne scheint hell. Der Mond scheint blass. Die Lampe scheint warm.
Das Kind spielt gerne. Der Hund spielt laut. Die Katze spielt leise.
Der Mann arbeitet viel. Die Frau arbeitet hart. Das Team arbeitet zusammen.
Die Blume waechst schnell. Der Baum waechst langsam. Das Gras waechst ueberall.
Der Lehrer lehrt gut. Die Schuelerin lernt fleissig. Das Buch lehrt viel.
Die Katze schlaeft lang. Der Hund schlaeft kurz. Das Baby schlaeft oft.
Der Fisch schwimmt tief. Die Ente schwimmt oben. Das Boot schwimmt ruhig.
Die Maus ist klein. Der Elefant ist gross. Die Ameise ist winzig.
Der Winter ist kalt. Der Sommer ist heiss. Der Fruehling ist mild.
Das Haus steht fest. Der Turm steht hoch. Die Huette steht schief.
Die Katze miaut laut. Der Hund bellt stark. Der Vogel singt schoen.
Der Vater liest viel. Die Mutter liest abends. Das Kind liest langsam.
Die Tasse ist voll. Das Glas ist leer. Der Teller ist sauber.
Der Berg ist steil. Das Tal ist flach. Der Huegel ist sanft.
"""


def tokenize(text):
    """Tokenisiert Text in Kleinbuchstaben-Woerter."""
    text = text.lower()
    for char in '.,!?;:()[]{}"\'-':
        text = text.replace(char, ' ')
    return text.split()


def compute_corpus_cooccurrence(vocabulary, window_size=2):
    """
    Berechnet tatsaechliche Ko-Okkurrenzen im Trainingskorpus.

    Args:
        vocabulary: word -> idx Mapping
        window_size: Kontextfenster-Groesse

    Returns:
        cooccurrence: Matrix (vocab_size, vocab_size) mit Ko-Okkurrenz-Zaehlungen
    """
    words = tokenize(TRAINING_CORPUS)
    vocab_size = len(vocabulary)
    cooccurrence = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for i, word in enumerate(words):
        if word not in vocabulary:
            continue
        target_idx = vocabulary[word]

        start = max(0, i - window_size)
        end = min(len(words), i + window_size + 1)

        for j in range(start, end):
            if i != j and words[j] in vocabulary:
                context_idx = vocabulary[words[j]]
                cooccurrence[target_idx, context_idx] += 1

    return cooccurrence


def categorize_absence_reason(target, context, corpus_count):
    """
    Kategorisiert warum ein Wort-Paar unter-trainiert ist.

    Args:
        target: Zielwort
        context: Kontextwort
        corpus_count: Wie oft treten sie zusammen im Korpus auf

    Returns:
        reason: String mit Kategorisierung
    """
    if target == context:
        return "Selbstreferenz (gleiches Wort)"
    if corpus_count == 0:
        return "Nie im Korpus zusammen"
    if corpus_count < 2:
        return "Seltene Ko-Okkurrenz (< 2)"
    return "Statistisches Under-Sampling"


def analyze_undertrained_linguistically(absence, cooccurrence, idx_to_word, vocabulary):
    """
    Linguistische Analyse der unter-trainierten Paare.

    Vergleicht Training-Absenz mit Korpus-Statistik um zu verstehen
    WARUM bestimmte Paare unter-trainiert wurden.

    Args:
        absence: Absenz-Matrix
        cooccurrence: Korpus Ko-Okkurrenz-Matrix
        idx_to_word: idx -> word Mapping
        vocabulary: word -> idx Mapping

    Returns:
        analysis: Liste von Dicts mit detaillierten Informationen
    """
    vocab_size = absence.shape[0]
    analysis = []

    flat_absence = absence.flatten()
    top_indices = np.argsort(flat_absence)[::-1][:20]

    for idx in top_indices:
        target = idx // vocab_size
        context = idx % vocab_size
        absence_score = flat_absence[idx]

        if absence_score <= 0:
            continue

        corpus_count = cooccurrence[target, context]
        target_word = idx_to_word.get(str(target), '?')
        context_word = idx_to_word.get(str(context), '?')

        analysis.append({
            'target': target_word,
            'context': context_word,
            'absence_score': float(absence_score),
            'corpus_cooccurrence': int(corpus_count),
            'reason': categorize_absence_reason(target_word, context_word, corpus_count)
        })

    return analysis


def build_pair_matrix(step_reader, vocab_size, max_steps=None):
    """
    Baut Matrix der tatsaechlich trainierten Paare.

    Args:
        step_reader: StepReader Instanz
        vocab_size: Groesse des Vokabulars
        max_steps: Maximale Schritte

    Returns:
        actual: Matrix (vocab_size, vocab_size) mit Zaehlungen
    """
    actual = np.zeros((vocab_size, vocab_size), dtype=np.int64)

    n = min(step_reader.num_steps, max_steps) if max_steps else step_reader.num_steps

    print(f"  Zaehle Trainingspaare ({n:,} Schritte)...")
    for i in range(n):
        step = step_reader.read_step(i)
        target = step['target_idx']
        context = step['context_idx']
        actual[target, context] += 1

        if (i + 1) % 100000 == 0:
            print(f"    {i+1:,} / {n:,}")

    return actual


def compute_absence_matrix(actual):
    """
    Berechnet Absenz-Matrix: Was wurde weniger trainiert als erwartet?

    Args:
        actual: Matrix der tatsaechlichen Trainings

    Returns:
        absence: Matrix mit Absenz-Scores (positiv = unter-trainiert)
    """
    # Theoretische Erwartung: Gleichverteilung
    total = actual.sum()
    n_pairs = actual.size
    expected = total / n_pairs

    # Absenz = Erwartung - Tatsaechlich
    absence = expected - actual

    # Nur positive Werte (unter-trainierte Paare)
    absence = np.maximum(absence, 0)

    return absence


def find_persistent_holes(step_reader, vocab_size, checkpoints=[100000, 200000, 500000]):
    """
    Findet "Loecher" die ueber verschiedene Trainingsphasen persistieren.

    Returns:
        persistent_holes: Liste von (target, context) Paaren die immer fehlen
        hole_evolution: Dict mapping checkpoint -> absence matrix
    """
    hole_evolution = {}

    for checkpoint in checkpoints:
        if checkpoint > step_reader.num_steps:
            continue

        actual = build_pair_matrix(step_reader, vocab_size, checkpoint)
        absence = compute_absence_matrix(actual)
        hole_evolution[checkpoint] = absence

    # Finde persistente Loecher (in allen Checkpoints abwesend)
    persistent_holes = []

    if len(hole_evolution) > 1:
        # Paare die in ALLEN Checkpoints hohe Absenz haben
        all_checkpoints = list(hole_evolution.values())
        combined = np.minimum.reduce(all_checkpoints)  # Minimum ueber alle

        # Finde Paare mit hoher kombinierter Absenz
        threshold = np.percentile(combined[combined > 0], 90) if np.any(combined > 0) else 0
        holes = np.argwhere(combined > threshold)

        for target, context in holes:
            persistent_holes.append((int(target), int(context)))

    return persistent_holes, hole_evolution


def analyze_hole_connectivity(absence, threshold_percentile=90):
    """
    Analysiert ob Trainings-Loecher zusammenhaengend oder verstreut sind.

    Returns:
        connectivity: Dict mit Konnektivitaets-Metriken
    """
    # Binarisiere Absenz-Matrix
    threshold = np.percentile(absence[absence > 0], threshold_percentile) if np.any(absence > 0) else 0
    binary_holes = (absence > threshold).astype(int)

    # Anzahl Loecher
    n_holes = binary_holes.sum()

    # Zeilen/Spalten mit Loechern
    rows_with_holes = np.sum(binary_holes.sum(axis=1) > 0)
    cols_with_holes = np.sum(binary_holes.sum(axis=0) > 0)

    # Cluster-Analyse (vereinfacht: zusammenhaengende Komponenten)
    # Hier: Zaehle wie viele isolierte vs. verbundene Loecher

    connectivity = {
        'n_holes': int(n_holes),
        'rows_with_holes': int(rows_with_holes),
        'cols_with_holes': int(cols_with_holes),
        'density': float(n_holes / binary_holes.size),
        'row_coverage': float(rows_with_holes / binary_holes.shape[0]),
        'col_coverage': float(cols_with_holes / binary_holes.shape[1])
    }

    return connectivity


def analyze_word_absence(actual, idx_to_word):
    """
    Analysiert welche Woerter besonders unter-trainiert sind.

    Returns:
        word_stats: Dict mapping word -> training stats
    """
    # Pro Wort: Wie oft als Target vs Context trainiert?
    target_counts = actual.sum(axis=1)  # Zeilen-Summe
    context_counts = actual.sum(axis=0)  # Spalten-Summe

    word_stats = {}
    for idx in range(len(target_counts)):
        word = idx_to_word.get(str(idx), f"word_{idx}")
        word_stats[word] = {
            'as_target': int(target_counts[idx]),
            'as_context': int(context_counts[idx]),
            'total': int(target_counts[idx] + context_counts[idx]),
            'target_context_ratio': float(target_counts[idx] / (context_counts[idx] + 1))
        }

    return word_stats


def correlate_holes_with_embeddings(absence, embedding_similarities):
    """
    Korreliert Trainings-Loecher mit Embedding-Aehnlichkeiten.

    Hypothese: Aehnliche Woerter werden vielleicht weniger oft zusammen trainiert?
    """
    # Flatten und korrelieren
    absence_flat = absence.flatten()
    sim_flat = embedding_similarities.flatten()

    # Nur Paare mit positiver Absenz
    mask = absence_flat > 0
    if mask.sum() > 10:
        correlation = np.corrcoef(absence_flat[mask], sim_flat[mask])[0, 1]
    else:
        correlation = 0

    return correlation


def visualize_results(actual, absence, persistent_holes, hole_evolution,
                      connectivity, word_stats, idx_to_word, output_dir):
    """Erstellt Visualisierungen der AC-Analyse."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    vocab_size = actual.shape[0]
    words = [idx_to_word.get(str(i), f"w{i}") for i in range(vocab_size)]

    # 1. Tatsaechliche Trainingsmatrix
    ax = axes[0, 0]
    im = ax.imshow(np.log1p(actual), cmap='Blues', aspect='auto')
    ax.set_xlabel('Context Word')
    ax.set_ylabel('Target Word')
    ax.set_title('Trainings-Haeufigkeit (log-Skala)')
    ax.set_xticks(range(0, vocab_size, 5))
    ax.set_yticks(range(0, vocab_size, 5))
    plt.colorbar(im, ax=ax)

    # 2. Absenz-Matrix
    ax = axes[0, 1]
    im = ax.imshow(absence, cmap='Reds', aspect='auto')
    ax.set_xlabel('Context Word')
    ax.set_ylabel('Target Word')
    ax.set_title('Absenz-Matrix (unter-trainierte Paare)')
    ax.set_xticks(range(0, vocab_size, 5))
    ax.set_yticks(range(0, vocab_size, 5))
    plt.colorbar(im, ax=ax)

    # 3. Trainings-Bias pro Wort
    ax = axes[0, 2]
    sorted_words = sorted(word_stats.items(), key=lambda x: x[1]['total'], reverse=True)
    word_names = [w for w, _ in sorted_words]
    target_vals = [s['as_target'] for _, s in sorted_words]
    context_vals = [s['as_context'] for _, s in sorted_words]

    x = np.arange(len(word_names))
    width = 0.35
    ax.barh(x - width/2, target_vals, width, label='Als Target', color='blue', alpha=0.7)
    ax.barh(x + width/2, context_vals, width, label='Als Context', color='green', alpha=0.7)
    ax.set_yticks(x)
    ax.set_yticklabels(word_names, fontsize=7)
    ax.set_xlabel('Haeufigkeit')
    ax.set_title('Training pro Wort')
    ax.legend()
    ax.invert_yaxis()

    # 4. Loch-Evolution
    ax = axes[1, 0]
    if len(hole_evolution) > 1:
        checkpoints = sorted(hole_evolution.keys())
        mean_absence = [hole_evolution[c].mean() for c in checkpoints]
        max_absence = [hole_evolution[c].max() for c in checkpoints]

        ax.plot(checkpoints, mean_absence, 'b-o', label='Mittlere Absenz')
        ax.plot(checkpoints, max_absence, 'r-o', label='Max Absenz')
        ax.set_xlabel('Trainingsschritt')
        ax.set_ylabel('Absenz')
        ax.set_title('Evolution der Trainings-Loecher')
        ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Persistente Loecher
    ax = axes[1, 1]
    if persistent_holes:
        hole_matrix = np.zeros_like(actual, dtype=float)
        for t, c in persistent_holes:
            hole_matrix[t, c] = 1
        im = ax.imshow(hole_matrix, cmap='Greys', aspect='auto')
        ax.set_xlabel('Context Word')
        ax.set_ylabel('Target Word')
        ax.set_title(f'Persistente Loecher ({len(persistent_holes)} Paare)')
    else:
        ax.text(0.5, 0.5, 'Keine persistenten Loecher gefunden',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Persistente Loecher')

    # 6. Zusammenfassung
    ax = axes[1, 2]
    ax.axis('off')

    # Top unter-trainierte Paare
    flat_absence = absence.flatten()
    top_indices = np.argsort(flat_absence)[::-1][:10]
    top_holes = []
    for idx in top_indices:
        target = idx // vocab_size
        context = idx % vocab_size
        value = flat_absence[idx]
        if value > 0:
            top_holes.append((idx_to_word.get(str(target), '?'),
                              idx_to_word.get(str(context), '?'),
                              value))

    summary = f"""
    ABSENCE CARTOGRAPHY - ZUSAMMENFASSUNG

    TRAININGS-STATISTIK:
    - Gesamte Trainings: {actual.sum():,}
    - Unique Paare trainiert: {(actual > 0).sum():,} / {actual.size}
    - Abdeckung: {(actual > 0).sum() / actual.size * 100:.1f}%

    LOCH-ANALYSE:
    - Loecher (Top 10%): {connectivity['n_holes']}
    - Zeilen betroffen: {connectivity['rows_with_holes']} / {vocab_size}
    - Spalten betroffen: {connectivity['cols_with_holes']} / {vocab_size}
    - Persistente Loecher: {len(persistent_holes)}

    TOP 5 UNTER-TRAINIERTE PAARE:
    {chr(10).join(f'  {t} -> {c}: {v:.1f}' for t, c, v in top_holes[:5])}

    INTERPRETATION:
    - Hohe Absenz bei bestimmten Paaren =
      Diese Kombinationen werden systematisch vermieden
    - Persistente Loecher = Strukturelle Luecken im Training
    """
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'ac_analysis.png')
    plt.savefig(output_path, dpi=150)
    print(f"  Gespeichert: {output_path}")
    plt.close()


def run_analysis(max_steps=None):
    """
    Fuehrt die vollstaendige AC-Analyse durch.
    """
    output_dir = ensure_output_dir()
    print("=" * 70)
    print("ABSENCE CARTOGRAPHY (AC) ANALYSE")
    print("=" * 70)

    vocab_data = load_vocabulary()
    vocabulary = vocab_data['vocabulary']
    idx_to_word = vocab_data['idx_to_word']
    vocab_size = len(vocabulary)

    print(f"\n  Vokabular: {vocab_size} Woerter")
    print(f"  Theoretische Paare: {vocab_size * vocab_size}")

    with StepReader(STEPS_FILE) as reader:
        # 1. Trainingsmatrix bauen
        actual = build_pair_matrix(reader, vocab_size, max_steps)
        print(f"\n  Trainierte Paare: {(actual > 0).sum()} / {actual.size}")

        # 2. Absenz-Matrix
        absence = compute_absence_matrix(actual)
        print(f"  Unter-trainierte Paare (Absenz > 0): {(absence > 0).sum()}")

        # 3. Persistente Loecher
        print("\n  Suche persistente Loecher...")
        n = reader.num_steps if max_steps is None else min(reader.num_steps, max_steps)
        checkpoints = [n // 4, n // 2, 3 * n // 4, n]
        checkpoints = [c for c in checkpoints if c > 0]
        persistent_holes, hole_evolution = find_persistent_holes(
            reader, vocab_size, checkpoints
        )
        print(f"  Persistente Loecher: {len(persistent_holes)}")

        # 4. Konnektivitaet
        connectivity = analyze_hole_connectivity(absence)
        print(f"\n  Loch-Konnektivitaet:")
        print(f"    Loecher (Top 10%): {connectivity['n_holes']}")
        print(f"    Zeilen-Abdeckung: {connectivity['row_coverage']*100:.1f}%")
        print(f"    Spalten-Abdeckung: {connectivity['col_coverage']*100:.1f}%")

        # 5. Wort-Statistiken
        word_stats = analyze_word_absence(actual, idx_to_word)

        # Top unter-trainierte Woerter
        sorted_by_total = sorted(word_stats.items(), key=lambda x: x[1]['total'])
        print(f"\n  Am wenigsten trainierte Woerter:")
        for word, stats in sorted_by_total[:5]:
            print(f"    {word}: {stats['total']} (T={stats['as_target']}, C={stats['as_context']})")

        # 6. Linguistische Korpus-Analyse
        print("\n  Vergleiche mit Korpus-Statistik...")
        cooccurrence = compute_corpus_cooccurrence(vocabulary)
        linguistic_analysis = analyze_undertrained_linguistically(
            absence, cooccurrence, idx_to_word, vocabulary
        )
        print(f"  Top unter-trainierte Paare mit Gruenden:")
        for item in linguistic_analysis[:5]:
            print(f"    {item['target']} -> {item['context']}: {item['reason']}")
            print(f"      (Korpus: {item['corpus_cooccurrence']}x zusammen)")

        # 7. Visualisierungen
        print("\n  Erstelle Visualisierungen...")
        visualize_results(actual, absence, persistent_holes, hole_evolution,
                          connectivity, word_stats, idx_to_word, output_dir)

    # Ergebnisse
    results = {
        'total_trainings': int(actual.sum()),
        'unique_pairs': int((actual > 0).sum()),
        'coverage': float((actual > 0).sum() / actual.size),
        'persistent_holes': persistent_holes[:100],
        'connectivity': connectivity,
        'word_stats': word_stats,
        'linguistic_analysis': linguistic_analysis
    }

    print("\n" + "=" * 70)
    print("AC ANALYSE ABGESCHLOSSEN")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_analysis()
