"""
Methode 18: Training-Specific Phenomena Detector (TSPD)
=======================================================

Konzept: Phaenomene finden die NUR im Kontext von NN-Training existieren.
Diese Effekte haben keine Analogie in klassischer Physik oder Statistik.

Ansatz:
- Interferenz-Muster: Wenn Woerter gemeinsamen Kontext teilen, interferieren Updates
- Echo-Kammern: Geschlossene Schleifen (A trainiert B trainiert A)
- Gradient-Verschmutzung: Indirekte Effekte durch geteilte Kontexte
- Lern-Bursts: Ploetzlich koordinierte Aenderungen ueber mehrere Woerter

Frage beantwortet:
- Welche emergenten Phaenomene entstehen durch die Trainingsstruktur?
- Gibt es Feedback-Schleifen die Muster verstaerken?

Autor: Maximilian (MI-Forschung)
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

from .common import StepReader, STEPS_FILE, load_vocabulary
from .common.utils import ensure_output_dir


def build_training_pair_graph(step_reader, vocab_data, max_steps=None):
    """
    Baut einen Graphen der Trainingsbeziehungen.

    Returns:
        pair_counts: Dict {(target, context): count}
        word_pairs: Dict {word: [(context, count), ...]}
    """
    idx_to_word = vocab_data['idx_to_word']
    n = min(step_reader.num_steps, max_steps) if max_steps else step_reader.num_steps

    pair_counts = defaultdict(int)
    word_pairs = defaultdict(list)

    print(f"  Baue Trainings-Graph...")

    for i in range(n):
        step = step_reader.read_step(i)
        target_idx = step['target_idx']
        context_idx = step['context_idx']

        target_word = idx_to_word.get(str(target_idx), f"w{target_idx}")
        context_word = idx_to_word.get(str(context_idx), f"w{context_idx}")

        pair_counts[(target_word, context_word)] += 1

        if (i + 1) % 200000 == 0:
            print(f"    {i + 1:,} / {n:,}")

    # Konvertiere zu word_pairs
    for (target, context), count in pair_counts.items():
        word_pairs[target].append((context, count))

    return dict(pair_counts), dict(word_pairs)


def detect_echo_chambers(pair_counts, min_bidirectional=5):
    """
    Findet Echo-Kammern: Paare wo A->B und B->A beide trainiert werden.

    Returns:
        echo_pairs: Liste von {word_a, word_b, a_to_b, b_to_a, echo_strength}
    """
    echo_pairs = []

    processed = set()

    for (target, context), count in pair_counts.items():
        if (target, context) in processed or (context, target) in processed:
            continue

        reverse_count = pair_counts.get((context, target), 0)

        if count >= min_bidirectional and reverse_count >= min_bidirectional:
            echo_strength = min(count, reverse_count) / max(count, reverse_count)
            echo_pairs.append({
                'word_a': target,
                'word_b': context,
                'a_to_b': count,
                'b_to_a': reverse_count,
                'total': count + reverse_count,
                'echo_strength': float(echo_strength)
            })

        processed.add((target, context))

    echo_pairs.sort(key=lambda x: x['total'], reverse=True)
    return echo_pairs


def detect_interference_patterns(step_reader, vocab_data, window_size=100, max_steps=None):
    """
    Findet Interferenz-Muster: Wenn das gleiche Wort kurz hintereinander
    mit verschiedenen Kontexten trainiert wird.

    Returns:
        interference_events: Liste von {word, step, contexts, variance}
    """
    idx_to_word = vocab_data['idx_to_word']
    n = min(step_reader.num_steps, max_steps) if max_steps else step_reader.num_steps

    interference_events = []
    word_recent_contexts = defaultdict(list)

    print(f"  Detektiere Interferenz-Muster...")

    for i in range(n):
        step = step_reader.read_step(i)
        target_idx = step['target_idx']
        context_idx = step['context_idx']
        change = step['change']

        target_word = idx_to_word.get(str(target_idx), f"w{target_idx}")

        # Fuege zum Recent-Buffer hinzu
        word_recent_contexts[target_word].append({
            'step': i,
            'context_idx': context_idx,
            'change': change
        })

        # Behalte nur letzten window_size Eintraege
        if len(word_recent_contexts[target_word]) > window_size:
            word_recent_contexts[target_word].pop(0)

        # Pruefe auf Interferenz (verschiedene Kontexte im Fenster)
        if len(word_recent_contexts[target_word]) >= 10:
            recent = word_recent_contexts[target_word]
            context_set = set(r['context_idx'] for r in recent)

            # Interferenz = viele verschiedene Kontexte
            if len(context_set) >= 5:
                # Berechne Varianz der Changes
                changes = np.array([r['change'] for r in recent])
                change_variance = np.var(changes)

                if change_variance > 0.0001:  # Signifikante Varianz
                    interference_events.append({
                        'word': target_word,
                        'step': i,
                        'n_contexts': len(context_set),
                        'change_variance': float(change_variance)
                    })

        if (i + 1) % 200000 == 0:
            print(f"    {i + 1:,} / {n:,}")

    return interference_events


def detect_learning_bursts(step_reader, vocab_data, window_size=50, threshold=2.0, max_steps=None):
    """
    Findet Lern-Bursts: Momente wo viele Woerter gleichzeitig
    ueberproportional grosse Aenderungen erfahren.

    Returns:
        bursts: Liste von {step, n_words, mean_magnitude, words_affected}
    """
    idx_to_word = vocab_data['idx_to_word']
    n = min(step_reader.num_steps, max_steps) if max_steps else step_reader.num_steps

    bursts = []

    # Sammle Magnitudes pro Fenster
    window_data = []

    print(f"  Detektiere Lern-Bursts...")

    for i in range(n):
        step = step_reader.read_step(i)
        target_idx = step['target_idx']
        word = idx_to_word.get(str(target_idx), f"w{target_idx}")
        magnitude = step['change_magnitude']

        window_data.append({
            'word': word,
            'magnitude': magnitude,
            'step': i
        })

        # Analysiere Fenster
        if len(window_data) >= window_size:
            magnitudes = [d['magnitude'] for d in window_data]
            mean_mag = np.mean(magnitudes)
            std_mag = np.std(magnitudes) + 1e-10

            # Finde ueberdurchschnittliche Steps
            high_mag_count = sum(1 for m in magnitudes if m > mean_mag + threshold * std_mag)

            if high_mag_count >= window_size * 0.3:  # 30% der Steps sind hoch
                affected_words = set(d['word'] for d in window_data
                                     if d['magnitude'] > mean_mag + threshold * std_mag)
                bursts.append({
                    'step': i,
                    'n_high_magnitude': high_mag_count,
                    'mean_magnitude': float(mean_mag),
                    'std_magnitude': float(std_mag),
                    'words_affected': list(affected_words)[:10]  # Top 10
                })

            # Slide window
            window_data.pop(0)

        if (i + 1) % 200000 == 0:
            print(f"    {i + 1:,} / {n:,}")

    return bursts


def analyze_gradient_contamination(step_reader, vocab_data, max_steps=None):
    """
    Analysiert Gradient-Verschmutzung: Wie stark beeinflusst ein Update
    eines Wortes andere Woerter indirekt?

    Proxy: Korrelation zwischen Changes von Woertern die gemeinsame
    Kontexte teilen.

    Returns:
        contamination_matrix: Korrelationsmatrix der Wort-Changes
        word_list: Liste der Woerter
    """
    idx_to_word = vocab_data['idx_to_word']
    n = min(step_reader.num_steps, max_steps) if max_steps else step_reader.num_steps

    # Sammle Changes pro Wort
    word_changes = defaultdict(list)

    print(f"  Analysiere Gradient-Verschmutzung...")

    for i in range(n):
        step = step_reader.read_step(i)
        target_idx = step['target_idx']
        word = idx_to_word.get(str(target_idx), f"w{target_idx}")
        change = step['change']

        word_changes[word].append(change)

        if (i + 1) % 200000 == 0:
            print(f"    {i + 1:,} / {n:,}")

    # Berechne durchschnittliche Change-Richtung pro Wort
    word_directions = {}
    word_list = list(word_changes.keys())

    for word in word_list:
        changes = np.array(word_changes[word])
        mean_direction = np.mean(changes, axis=0)
        norm = np.linalg.norm(mean_direction)
        if norm > 1e-10:
            word_directions[word] = mean_direction / norm
        else:
            word_directions[word] = mean_direction

    # Korrelationsmatrix der Richtungen
    n_words = len(word_list)
    contamination_matrix = np.zeros((n_words, n_words))

    for i, word_i in enumerate(word_list):
        for j, word_j in enumerate(word_list):
            if i != j:
                dir_i = word_directions[word_i]
                dir_j = word_directions[word_j]
                contamination_matrix[i, j] = float(np.dot(dir_i, dir_j))

    return contamination_matrix, word_list


def detect_coordinated_movements(step_reader, vocab_data, window_size=100, max_steps=None):
    """
    Findet koordinierte Bewegungen: Momente wo mehrere Woerter
    in die gleiche Richtung driften.

    Returns:
        coordinated_events: Liste von {step, alignment_score, words_involved}
    """
    idx_to_word = vocab_data['idx_to_word']
    n = min(step_reader.num_steps, max_steps) if max_steps else step_reader.num_steps

    coordinated_events = []
    window_changes = defaultdict(list)

    print(f"  Detektiere koordinierte Bewegungen...")

    for i in range(n):
        step = step_reader.read_step(i)
        target_idx = step['target_idx']
        word = idx_to_word.get(str(target_idx), f"w{target_idx}")
        change = step['change']

        window_changes[word].append(change)

        # Analysiere alle window_size Schritte
        if i > 0 and i % window_size == 0:
            # Berechne durchschnittliche Richtung pro Wort
            directions = {}
            for w, changes in window_changes.items():
                if len(changes) >= 3:
                    mean_change = np.mean(changes, axis=0)
                    norm = np.linalg.norm(mean_change)
                    if norm > 1e-10:
                        directions[w] = mean_change / norm

            # Berechne paarweise Alignment
            if len(directions) >= 3:
                words = list(directions.keys())
                alignments = []
                for i_w in range(len(words)):
                    for j_w in range(i_w + 1, len(words)):
                        alignment = np.dot(directions[words[i_w]], directions[words[j_w]])
                        alignments.append(alignment)

                mean_alignment = np.mean(alignments)

                if mean_alignment > 0.5:  # Stark koordiniert
                    coordinated_events.append({
                        'step': i,
                        'alignment_score': float(mean_alignment),
                        'n_words': len(directions),
                        'words_involved': list(directions.keys())[:10]
                    })

            # Reset window
            window_changes = defaultdict(list)

        if (i + 1) % 200000 == 0:
            print(f"    {i + 1:,} / {n:,}")

    return coordinated_events


def visualize_results(echo_pairs, interference_events, bursts, contamination_matrix,
                      word_list, coordinated_events, output_dir):
    """Erstellt Visualisierungen."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Echo-Kammern
    ax = axes[0, 0]
    if echo_pairs:
        top_echoes = echo_pairs[:15]
        labels = [f"{e['word_a']}-{e['word_b']}" for e in top_echoes]
        totals = [e['total'] for e in top_echoes]
        strengths = [e['echo_strength'] for e in top_echoes]

        x = range(len(labels))
        ax.bar(x, totals, alpha=0.7, label='Total')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Trainingspaare')
        ax.set_title(f'Top Echo-Kammern ({len(echo_pairs)} gefunden)')

    # 2. Interferenz-Muster ueber Zeit
    ax = axes[0, 1]
    if interference_events:
        steps = [e['step'] for e in interference_events]
        variances = [e['change_variance'] for e in interference_events]
        ax.scatter(steps, variances, alpha=0.5, s=10)
        ax.set_xlabel('Trainingsschritt')
        ax.set_ylabel('Change Variance')
        ax.set_title(f'Interferenz-Events ({len(interference_events)} Events)')
        ax.set_yscale('log')

    # 3. Lern-Bursts Timeline
    ax = axes[0, 2]
    if bursts:
        steps = [b['step'] for b in bursts]
        n_high = [b['n_high_magnitude'] for b in bursts]
        ax.scatter(steps, n_high, alpha=0.5)
        ax.set_xlabel('Trainingsschritt')
        ax.set_ylabel('High-Magnitude Steps')
        ax.set_title(f'Lern-Bursts ({len(bursts)} Bursts)')

    # 4. Gradient-Verschmutzung Matrix
    ax = axes[1, 0]
    if contamination_matrix is not None and len(contamination_matrix) > 0:
        im = ax.imshow(contamination_matrix, cmap='RdBu_r', aspect='auto',
                       vmin=-1, vmax=1)
        ax.set_xlabel('Wort Index')
        ax.set_ylabel('Wort Index')
        ax.set_title('Gradient-Verschmutzung (Richtungs-Korrelation)')
        plt.colorbar(im, ax=ax)

    # 5. Koordinierte Bewegungen
    ax = axes[1, 1]
    if coordinated_events:
        steps = [e['step'] for e in coordinated_events]
        alignments = [e['alignment_score'] for e in coordinated_events]
        ax.plot(steps, alignments, 'b-', alpha=0.7)
        ax.scatter(steps, alignments, alpha=0.5, s=20)
        ax.set_xlabel('Trainingsschritt')
        ax.set_ylabel('Alignment Score')
        ax.set_title(f'Koordinierte Bewegungen ({len(coordinated_events)} Events)')
        ax.axhline(0.5, color='r', linestyle='--', alpha=0.5)

    # 6. Zusammenfassung
    ax = axes[1, 2]
    ax.axis('off')

    n_echo = len(echo_pairs) if echo_pairs else 0
    n_interference = len(interference_events) if interference_events else 0
    n_bursts = len(bursts) if bursts else 0
    n_coordinated = len(coordinated_events) if coordinated_events else 0

    mean_contamination = np.mean(np.abs(contamination_matrix)) if contamination_matrix is not None else 0

    summary = f"""
    TRAINING-SPECIFIC PHENOMENA DETECTOR (TSPD)
    ==========================================

    ECHO-KAMMERN:
    - Bidirektionale Paare: {n_echo}
    - Top Echo: {echo_pairs[0]['word_a']}-{echo_pairs[0]['word_b']} ({echo_pairs[0]['total']}x) if echo_pairs else 'N/A'

    INTERFERENZ-MUSTER:
    - Events detektiert: {n_interference}
    - Woerter mit Interferenz: {len(set(e['word'] for e in interference_events)) if interference_events else 0}

    LERN-BURSTS:
    - Bursts detektiert: {n_bursts}
    - Durchschn. betroffene Woerter: {np.mean([len(b['words_affected']) for b in bursts]):.1f} if bursts else 0

    GRADIENT-VERSCHMUTZUNG:
    - Analysierte Woerter: {len(word_list) if word_list else 0}
    - Mean Abs. Korrelation: {mean_contamination:.4f}

    KOORDINIERTE BEWEGUNGEN:
    - Events detektiert: {n_coordinated}
    - Max Alignment: {max(e['alignment_score'] for e in coordinated_events):.3f} if coordinated_events else 0

    INTERPRETATION:
    - Echo-Kammern = Feedback-Schleifen im Training
    - Interferenz = Competing Updates fuer gleiches Wort
    - Bursts = Phasenuebergaenge/Reorganisation
    - Verschmutzung = Indirekte Update-Effekte
    - Koordination = Emergentes kollektives Verhalten
    """
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'tspd_analysis.png')
    plt.savefig(output_path, dpi=150)
    print(f"  Gespeichert: {output_path}")
    plt.close()


def run_analysis(max_steps=None, window_size=100):
    """
    Fuehrt die vollstaendige TSPD-Analyse durch.
    """
    output_dir = ensure_output_dir()

    print("=" * 70)
    print("TRAINING-SPECIFIC PHENOMENA DETECTOR (TSPD)")
    print("=" * 70)

    vocab_data = load_vocabulary()

    print(f"\n  Konfiguration:")
    print(f"    Window Size: {window_size}")

    with StepReader(STEPS_FILE) as reader:
        n = min(reader.num_steps, max_steps) if max_steps else reader.num_steps
        print(f"    Steps: {n:,}")

        # 1. Trainings-Graph
        print("\n[1/6] Baue Trainings-Graph...")
        pair_counts, word_pairs = build_training_pair_graph(reader, vocab_data, max_steps)
        print(f"  {len(pair_counts)} unique Paare")

        # 2. Echo-Kammern
        print("\n[2/6] Detektiere Echo-Kammern...")
        echo_pairs = detect_echo_chambers(pair_counts)
        print(f"  {len(echo_pairs)} Echo-Paare gefunden")

        # 3. Interferenz
        print("\n[3/6] Detektiere Interferenz-Muster...")
        interference_events = detect_interference_patterns(reader, vocab_data, window_size, max_steps)
        print(f"  {len(interference_events)} Interferenz-Events")

        # 4. Lern-Bursts
        print("\n[4/6] Detektiere Lern-Bursts...")
        bursts = detect_learning_bursts(reader, vocab_data, window_size, max_steps=max_steps)
        print(f"  {len(bursts)} Bursts gefunden")

        # 5. Gradient-Verschmutzung
        print("\n[5/6] Analysiere Gradient-Verschmutzung...")
        contamination_matrix, word_list = analyze_gradient_contamination(reader, vocab_data, max_steps)
        print(f"  {len(word_list)} Woerter analysiert")

        # 6. Koordinierte Bewegungen
        print("\n[6/6] Detektiere koordinierte Bewegungen...")
        coordinated_events = detect_coordinated_movements(reader, vocab_data, window_size, max_steps)
        print(f"  {len(coordinated_events)} koordinierte Events")

    # Visualisierungen
    print("\n  Erstelle Visualisierungen...")
    visualize_results(echo_pairs, interference_events, bursts, contamination_matrix,
                      word_list, coordinated_events, output_dir)

    # Ergebnisse
    results = {
        'n_training_pairs': len(pair_counts),
        'echo_chambers': {
            'n_pairs': len(echo_pairs),
            'top_pairs': echo_pairs[:10]
        },
        'interference': {
            'n_events': len(interference_events),
            'n_words_affected': len(set(e['word'] for e in interference_events)) if interference_events else 0
        },
        'bursts': {
            'n_bursts': len(bursts),
            'mean_words_per_burst': float(np.mean([len(b['words_affected']) for b in bursts])) if bursts else 0
        },
        'gradient_contamination': {
            'n_words': len(word_list),
            'mean_abs_correlation': float(np.mean(np.abs(contamination_matrix))) if contamination_matrix is not None else 0
        },
        'coordinated_movements': {
            'n_events': len(coordinated_events),
            'max_alignment': float(max(e['alignment_score'] for e in coordinated_events)) if coordinated_events else 0
        }
    }

    print("\n" + "=" * 70)
    print("TSPD ANALYSE ABGESCHLOSSEN")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_analysis()
