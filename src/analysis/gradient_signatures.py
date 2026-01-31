"""
Methode 1: Gradient Signature Sequences (GSS)
=============================================

Konzept: Ignoriere Gradient-Magnitude. Konvertiere jeden Gradienten in eine
binaere Signatur (nur Vorzeichen), analysiere als symbolische Dynamik.

Frage beantwortet:
- #5 (Reihenfolge statt Wert)
- #2 (episodische Einfluesse)

Autor: Maximilian (MI-Forschung)
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

from .common import StepReader, STEPS_FILE, load_vocabulary
from .common.utils import ensure_output_dir


def gradient_to_signature(grad):
    """
    Konvertiert 10D Gradient zu Integer-Signatur (0-1023).

    Jedes Bit repraesentiert das Vorzeichen einer Dimension:
    - Bit i = 1 wenn grad[i] > 0
    - Bit i = 0 wenn grad[i] <= 0
    """
    return sum(1 << i for i, g in enumerate(grad) if g > 0)


def signature_to_binary_string(sig, dim=10):
    """Konvertiert Signatur zu lesbarem Binaerstring."""
    return format(sig, f'0{dim}b')


def analyze_signatures(step_reader, max_steps=None):
    """
    Analysiert alle Gradienten-Signaturen.

    Returns:
        signatures: Liste aller Signaturen in Reihenfolge
        signature_counts: Dict mit Haeufigkeiten pro Signatur
    """
    print("Analysiere Gradient-Signaturen...")
    n = min(step_reader.num_steps, max_steps) if max_steps else step_reader.num_steps

    signatures = []
    signature_counts = defaultdict(int)

    for i in range(n):
        step = step_reader.read_step(i)
        sig = gradient_to_signature(step['gradient'])
        signatures.append(sig)
        signature_counts[sig] += 1

        if (i + 1) % 100000 == 0:
            print(f"  Verarbeitet: {i+1:,} / {n:,}")

    return signatures, dict(signature_counts)


def build_transition_matrix(signatures, n_states=1024):
    """
    Baut Uebergangsmatrix T[i,j] = P(Signatur j folgt auf Signatur i).

    Returns:
        T: Normalisierte Uebergangsmatrix (1024 x 1024)
        T_counts: Roh-Zaehlungen
    """
    print("Baue Transitionsmatrix...")
    T_counts = np.zeros((n_states, n_states), dtype=np.int64)

    for i in range(len(signatures) - 1):
        current = signatures[i]
        next_sig = signatures[i + 1]
        T_counts[current, next_sig] += 1

    # Normalisieren (Zeilen-Summe = 1)
    row_sums = T_counts.sum(axis=1, keepdims=True)
    T = np.divide(T_counts, row_sums, where=row_sums > 0, out=np.zeros_like(T_counts, dtype=float))

    return T, T_counts


def find_attractor_signatures(T, threshold=0.1):
    """
    Findet "Attraktorsignaturen" mit hoher Selbsttransition.

    Attraktoren: Signaturen die haeufig auf sich selbst folgen.
    """
    self_transitions = np.diag(T)
    attractors = np.where(self_transitions > threshold)[0]

    return [(sig, self_transitions[sig]) for sig in attractors]


def find_repeller_signatures(T, T_counts, min_occurrences=100):
    """
    Findet "Repeller-Signaturen" die nie auf sich selbst folgen.

    Repeller: Signaturen die immer zu anderen uebergehen.
    """
    # Nur Signaturen betrachten die oft genug vorkommen
    occurrences = T_counts.sum(axis=1)
    frequent = occurrences >= min_occurrences

    self_transitions = np.diag(T)
    repellers = np.where((self_transitions == 0) & frequent)[0]

    return [(sig, int(occurrences[sig])) for sig in repellers]


def find_forbidden_transitions(T_counts, min_source_occurrences=50):
    """
    Findet "verbotene Transitionen" - Uebergaenge die nie beobachtet wurden.

    Interessant: Paare (i, j) wo Signatur i haeufig vorkommt aber nie zu j wechselt.
    """
    source_occurrences = T_counts.sum(axis=1)
    frequent_sources = source_occurrences >= min_source_occurrences

    forbidden = []
    for i in range(len(T_counts)):
        if frequent_sources[i]:
            for j in range(len(T_counts)):
                if T_counts[i, j] == 0 and source_occurrences[j] > 0:
                    forbidden.append((i, j))

    return forbidden


def compute_transition_entropy(T):
    """
    Berechnet Entropie der Uebergaenge pro Signatur.

    Hohe Entropie = viele moegliche Nachfolger
    Niedrige Entropie = vorhersagbarer Nachfolger
    """
    entropy = np.zeros(len(T))
    for i in range(len(T)):
        probs = T[i]
        probs = probs[probs > 0]  # Nur positive Wahrscheinlichkeiten
        if len(probs) > 0:
            entropy[i] = -np.sum(probs * np.log2(probs))
    return entropy


def find_phase_boundary_signatures(signatures, signature_counts, window=100):
    """
    Findet Signaturen die als "Phasengrenzen" fungieren.

    Phasengrenze: Signatur die selten vorkommt, aber wenn sie auftritt,
    markiert sie einen Uebergang zwischen verschiedenen Regimes.
    """
    # Berechne lokale Verteilungsdifferenz vor/nach jeder Signatur
    phase_scores = defaultdict(list)

    for i in range(window, len(signatures) - window):
        sig = signatures[i]

        # Verteilung vor und nach
        before = signatures[i-window:i]
        after = signatures[i+1:i+1+window]

        # Wie unterschiedlich sind die Verteilungen?
        before_set = set(before)
        after_set = set(after)
        jaccard = len(before_set & after_set) / len(before_set | after_set)
        diff_score = 1 - jaccard  # Hoher Score = grosse Differenz

        phase_scores[sig].append(diff_score)

    # Durchschnittliche "Phasengrenz-Staerke" pro Signatur
    boundary_strength = {
        sig: np.mean(scores)
        for sig, scores in phase_scores.items()
        if len(scores) >= 10  # Mindestens 10 Beobachtungen
    }

    return sorted(boundary_strength.items(), key=lambda x: x[1], reverse=True)


def visualize_results(signature_counts, T, entropy, attractors, repellers,
                      boundaries, output_dir):
    """Erstellt Visualisierungen der GSS-Analyse."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Signatur-Haeufigkeitsverteilung
    ax = axes[0, 0]
    counts = sorted(signature_counts.values(), reverse=True)
    ax.plot(counts, 'b-', alpha=0.7)
    ax.set_xlabel('Rang')
    ax.set_ylabel('Haeufigkeit')
    ax.set_title('Signatur-Haeufigkeitsverteilung')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 2. Transitionsmatrix (Heatmap fuer haeufigste Signaturen)
    ax = axes[0, 1]
    top_sigs = sorted(signature_counts.keys(), key=signature_counts.get, reverse=True)[:50]
    T_sub = T[np.ix_(top_sigs, top_sigs)]
    im = ax.imshow(T_sub, cmap='viridis', aspect='auto')
    ax.set_title('Transitionsmatrix (Top 50 Signaturen)')
    ax.set_xlabel('Ziel-Signatur')
    ax.set_ylabel('Quell-Signatur')
    plt.colorbar(im, ax=ax)

    # 3. Selbsttransitions-Verteilung
    ax = axes[0, 2]
    self_trans = np.diag(T)
    active = self_trans[self_trans > 0]
    ax.hist(active, bins=50, color='green', alpha=0.7)
    ax.set_xlabel('Selbsttransitions-Wahrscheinlichkeit')
    ax.set_ylabel('Anzahl Signaturen')
    ax.set_title('Verteilung der Selbsttransitionen')
    ax.axvline(0.1, color='red', linestyle='--', label='Attraktor-Schwelle')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Entropie-Verteilung
    ax = axes[1, 0]
    active_entropy = entropy[entropy > 0]
    ax.hist(active_entropy, bins=50, color='orange', alpha=0.7)
    ax.set_xlabel('Transitions-Entropie (bits)')
    ax.set_ylabel('Anzahl Signaturen')
    ax.set_title('Entropie der Uebergaenge')
    ax.grid(True, alpha=0.3)

    # 5. Attraktor vs Repeller
    ax = axes[1, 1]
    attractor_sigs = [a[0] for a in attractors[:20]]
    attractor_strengths = [a[1] for a in attractors[:20]]
    repeller_sigs = [r[0] for r in repellers[:20]]
    repeller_occs = [r[1] for r in repellers[:20]]

    x = range(max(len(attractor_strengths), len(repeller_occs)))
    if attractor_strengths:
        ax.bar([i - 0.2 for i in range(len(attractor_strengths))],
               attractor_strengths, 0.4, label='Attraktoren', color='blue')
    if repeller_occs:
        # Normalisieren fuer Vergleichbarkeit
        max_occ = max(repeller_occs) if repeller_occs else 1
        ax.bar([i + 0.2 for i in range(len(repeller_occs))],
               [o/max_occ for o in repeller_occs], 0.4, label='Repeller (norm)', color='red')
    ax.set_xlabel('Rang')
    ax.set_ylabel('Staerke / Normalisierte Haeufigkeit')
    ax.set_title('Top 20 Attraktoren vs Repeller')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Phasengrenzen
    ax = axes[1, 2]
    if boundaries:
        boundary_sigs = [b[0] for b in boundaries[:30]]
        boundary_scores = [b[1] for b in boundaries[:30]]
        ax.barh(range(len(boundary_scores)), boundary_scores, color='purple', alpha=0.7)
        ax.set_yticks(range(len(boundary_scores)))
        ax.set_yticklabels([signature_to_binary_string(s) for s in boundary_sigs], fontsize=6)
        ax.set_xlabel('Phasengrenz-Staerke')
        ax.set_title('Top 30 Phasengrenz-Signaturen')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'gss_analysis.png')
    plt.savefig(output_path, dpi=150)
    print(f"  Gespeichert: {output_path}")
    plt.close()


def run_analysis(max_steps=None):
    """
    Fuehrt die vollstaendige GSS-Analyse durch.

    Args:
        max_steps: Maximal zu analysierende Schritte (None = alle)

    Returns:
        dict mit allen Ergebnissen
    """
    output_dir = ensure_output_dir()
    print("=" * 70)
    print("GRADIENT SIGNATURE SEQUENCES (GSS) ANALYSE")
    print("=" * 70)

    # Daten laden
    vocab_data = load_vocabulary()
    print(f"\nKonfiguration: {vocab_data['config']}")

    with StepReader(STEPS_FILE) as reader:
        # 1. Signaturen extrahieren
        signatures, signature_counts = analyze_signatures(reader, max_steps)

        print(f"\n  Gefundene unique Signaturen: {len(signature_counts)}")
        print(f"  Theoretisch moeglich: 1024")

        # 2. Transitionsmatrix bauen
        T, T_counts = build_transition_matrix(signatures)

        # 3. Attraktoren finden
        attractors = find_attractor_signatures(T, threshold=0.05)
        print(f"\n  Attraktorsignaturen (Selbsttransition > 5%): {len(attractors)}")
        for sig, strength in attractors[:5]:
            print(f"    {signature_to_binary_string(sig)}: {strength:.3f}")

        # 4. Repeller finden
        repellers = find_repeller_signatures(T, T_counts, min_occurrences=100)
        print(f"\n  Repeller-Signaturen (nie Selbsttransition): {len(repellers)}")
        for sig, occurrences in repellers[:5]:
            print(f"    {signature_to_binary_string(sig)}: {occurrences:,} Vorkommen")

        # 5. Verbotene Transitionen
        forbidden = find_forbidden_transitions(T_counts, min_source_occurrences=50)
        print(f"\n  Verbotene Transitionen: {len(forbidden)}")

        # 6. Entropie berechnen
        entropy = compute_transition_entropy(T)
        active_entropy = entropy[entropy > 0]
        print(f"\n  Mittlere Transitions-Entropie: {np.mean(active_entropy):.2f} bits")
        print(f"  Max Entropie: {np.max(active_entropy):.2f} bits")

        # 7. Phasengrenzen finden
        print("\n  Suche Phasengrenzen...")
        boundaries = find_phase_boundary_signatures(signatures, signature_counts)
        print(f"  Gefundene Phasengrenz-Signaturen: {len(boundaries)}")
        for sig, score in boundaries[:5]:
            print(f"    {signature_to_binary_string(sig)}: Score {score:.3f}")

        # 8. Visualisierung
        print("\n  Erstelle Visualisierungen...")
        visualize_results(signature_counts, T, entropy, attractors, repellers,
                          boundaries, output_dir)

    # Ergebnisse zusammenfassen
    results = {
        'num_unique_signatures': len(signature_counts),
        'signature_counts': signature_counts,
        'transition_matrix': T,
        'attractors': attractors,
        'repellers': repellers,
        'forbidden_transitions_count': len(forbidden),
        'mean_entropy': float(np.mean(active_entropy)),
        'max_entropy': float(np.max(active_entropy)),
        'phase_boundaries': boundaries[:50]
    }

    print("\n" + "=" * 70)
    print("GSS ANALYSE ABGESCHLOSSEN")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_analysis(max_steps=None)
