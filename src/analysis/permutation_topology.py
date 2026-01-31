"""
Methode 8: Permutation Topology Analysis (PTA)
==============================================

Konzept: Behandle Schritte als Punkte im Feature-Raum.
Finde natuerliche Ordnungen durch Nearest-Neighbor-Ketten,
vergleiche mit zeitlicher Ordnung.

Frage beantwortet:
- #1 (alternative Ordnungen)

Hypothese: "Archetypische Trainingsmomente" wiederholen sich ueber Runs
hinweg, unabhaengig von Zeit.

Autor: Maximilian (MI-Forschung)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.stats import kendalltau, spearmanr
import os

from .common import StepReader, STEPS_FILE, load_vocabulary
from .common.utils import ensure_output_dir


def extract_step_features(step):
    """
    Extrahiert Feature-Vektor aus einem Schritt.

    Features:
    - loss
    - change_magnitude
    - gradient_norm
    - gradient_entropy (wie gleichmaessig verteilt)
    - change_direction_entropy
    """
    gradient = step['gradient']
    change = step['change']

    grad_norm = np.linalg.norm(gradient)

    # Gradient-Entropie (basierend auf normalisierten Absolutwerten)
    abs_grad = np.abs(gradient)
    if abs_grad.sum() > 0:
        probs = abs_grad / abs_grad.sum()
        grad_entropy = -np.sum(probs * np.log2(probs + 1e-10))
    else:
        grad_entropy = 0

    return np.array([
        step['loss'],
        step['change_magnitude'],
        grad_norm,
        grad_entropy
    ])


def build_feature_matrix(step_reader, sample_every=100, max_steps=None):
    """
    Baut Feature-Matrix aus gesampelten Schritten.

    Returns:
        features: Array (N, n_features)
        sampled_indices: Originale Schritt-Indizes
    """
    n = min(step_reader.num_steps, max_steps) if max_steps else step_reader.num_steps
    sampled_indices = list(range(0, n, sample_every))

    print(f"  Extrahiere Features fuer {len(sampled_indices)} Schritte...")

    features = []
    for i, idx in enumerate(sampled_indices):
        step = step_reader.read_step(idx)
        feat = extract_step_features(step)
        features.append(feat)

        if (i + 1) % 1000 == 0:
            print(f"    {i+1:,} / {len(sampled_indices):,}")

    features = np.array(features)

    # Normalisieren
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-10
    features = (features - mean) / std

    return features, np.array(sampled_indices)


def find_greedy_path(features, start=0, max_neighbors=50):
    """
    Findet Greedy-Pfad durch den Feature-Raum.

    Startet bei `start`, geht immer zum naechsten unbesuchten Nachbarn.

    Returns:
        path: Liste von Indizes in der "natuerlichen" Ordnung
    """
    n = len(features)
    nn = NearestNeighbors(n_neighbors=min(max_neighbors, n))
    nn.fit(features)

    visited = set()
    path = [start]
    visited.add(start)
    current = start

    while len(path) < n:
        # Finde naechste Nachbarn
        _, neighbors = nn.kneighbors([features[current]], n_neighbors=min(max_neighbors, n))
        neighbors = neighbors[0]

        # Gehe zum ersten unbesuchten
        found = False
        for neighbor in neighbors:
            if neighbor not in visited:
                path.append(neighbor)
                visited.add(neighbor)
                current = neighbor
                found = True
                break

        if not found:
            # Kein Nachbar verfuegbar - springe zum naechsten unbesuchten
            for i in range(n):
                if i not in visited:
                    path.append(i)
                    visited.add(i)
                    current = i
                    break

    return path


def compute_permutation_distance(path):
    """
    Berechnet verschiedene Masse fuer die "Permutations-Distanz".

    Vergleicht gefundene Ordnung mit der zeitlichen Ordnung [0, 1, 2, ...].

    Returns:
        metrics: Dict mit verschiedenen Distanzmassen
    """
    n = len(path)
    temporal_order = np.arange(n)
    natural_order = np.array(path)

    # Korrelationen
    pearson = np.corrcoef(natural_order, temporal_order)[0, 1]
    kendall, _ = kendalltau(natural_order, temporal_order)
    spearman, _ = spearmanr(natural_order, temporal_order)

    # Anzahl Inversionen (wie viele Paare sind "falsch" sortiert)
    inversions = 0
    for i in range(n):
        for j in range(i + 1, n):
            if path[i] > path[j]:
                inversions += 1
    max_inversions = n * (n - 1) // 2
    inversion_ratio = inversions / max_inversions

    return {
        'pearson': pearson,
        'kendall': kendall,
        'spearman': spearman,
        'inversion_ratio': inversion_ratio
    }


def find_archetypal_clusters(features, n_clusters=10):
    """
    Findet "archetypische" Trainingsmomente durch Clustering.

    Returns:
        labels: Cluster-Labels pro Schritt
        centers: Cluster-Zentren
    """
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    centers = kmeans.cluster_centers_

    return labels, centers


def analyze_cluster_timing(labels, sampled_indices, steps_per_epoch=360):
    """
    Analysiert wann Cluster-Mitglieder zeitlich auftreten.

    Returns:
        cluster_timing: Dict mapping cluster -> timing statistics
    """
    cluster_timing = {}

    for cluster_id in range(max(labels) + 1):
        member_indices = sampled_indices[labels == cluster_id]

        # Epochen der Mitglieder
        epochs = member_indices // steps_per_epoch

        cluster_timing[cluster_id] = {
            'count': len(member_indices),
            'mean_epoch': float(np.mean(epochs)),
            'std_epoch': float(np.std(epochs)),
            'min_epoch': int(np.min(epochs)),
            'max_epoch': int(np.max(epochs)),
            'span': int(np.max(epochs) - np.min(epochs))
        }

    return cluster_timing


def visualize_results(features, path, metrics, labels, cluster_timing,
                      sampled_indices, output_dir):
    """Erstellt Visualisierungen der PTA-Analyse."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Feature-Raum (PCA 2D)
    ax = axes[0, 0]
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    # Faerbe nach zeitlicher Ordnung
    colors = np.arange(len(features))
    scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1],
                         c=colors, cmap='viridis', alpha=0.5, s=10)
    plt.colorbar(scatter, ax=ax, label='Zeitlicher Index')
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_title('Feature-Raum (gefaerbt nach Zeit)')

    # 2. Greedy-Pfad visualisiert
    ax = axes[0, 1]
    path_order = np.arange(len(path))
    scatter = ax.scatter(features_2d[path, 0], features_2d[path, 1],
                         c=path_order, cmap='plasma', alpha=0.5, s=10)
    # Verbinde einige Punkte
    for i in range(0, len(path) - 1, len(path) // 100):
        ax.plot([features_2d[path[i], 0], features_2d[path[i+1], 0]],
                [features_2d[path[i], 1], features_2d[path[i+1], 1]],
                'k-', alpha=0.1, linewidth=0.5)
    plt.colorbar(scatter, ax=ax, label='Pfad-Position')
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_title('Greedy-Pfad durch Feature-Raum')

    # 3. Natuerliche vs Zeitliche Ordnung
    ax = axes[0, 2]
    ax.scatter(np.arange(len(path)), path, alpha=0.3, s=5)
    ax.plot([0, len(path)], [0, len(path)], 'r--', label='Perfekte Korrelation')
    ax.set_xlabel('Position im Greedy-Pfad')
    ax.set_ylabel('Originaler Zeitindex')
    ax.set_title(f'Ordnungsvergleich (Kendall={metrics["kendall"]:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Cluster im Feature-Raum
    ax = axes[1, 0]
    scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1],
                         c=labels, cmap='tab10', alpha=0.5, s=10)
    plt.colorbar(scatter, ax=ax, label='Cluster')
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_title('Archetypische Cluster')

    # 5. Cluster-Timing
    ax = axes[1, 1]
    clusters = sorted(cluster_timing.keys())
    spans = [cluster_timing[c]['span'] for c in clusters]
    counts = [cluster_timing[c]['count'] for c in clusters]

    ax.bar(clusters, spans, color='steelblue', alpha=0.7, label='Epoch-Span')
    ax2 = ax.twinx()
    ax2.plot(clusters, counts, 'ro-', label='Anzahl')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Epoch-Span')
    ax2.set_ylabel('Anzahl Mitglieder')
    ax.set_title('Cluster: Zeitliche Ausdehnung vs Groesse')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 6. Zusammenfassung
    ax = axes[1, 2]
    ax.axis('off')

    # Cluster mit groesstem Span (= zeitlich am meisten verteilt)
    max_span_cluster = max(cluster_timing.keys(), key=lambda c: cluster_timing[c]['span'])

    summary = f"""
    PERMUTATION TOPOLOGY ANALYSIS - ZUSAMMENFASSUNG

    ORDNUNGS-METRIKEN:
    - Pearson Korrelation: {metrics['pearson']:.3f}
    - Kendall Tau: {metrics['kendall']:.3f}
    - Spearman Rho: {metrics['spearman']:.3f}
    - Inversions-Ratio: {metrics['inversion_ratio']:.3f}

    INTERPRETATION:
    - Hohe Korrelation = Natuerliche ~ Zeitliche Ordnung
    - Niedrige Korrelation = Topologische Ordnung anders als Zeit

    ARCHETYPISCHE CLUSTER:
    - Anzahl Cluster: {len(cluster_timing)}
    - Groesster Cluster: {max(counts)} Schritte
    - Cluster mit groesstem Span: {max_span_cluster}
      (Span: {cluster_timing[max_span_cluster]['span']} Epochen)

    BEDEUTUNG:
    - Cluster die ueber viele Epochen verteilt sind =
      "Archetypische Momente" die sich wiederholen
    - Geringe Korrelation = Zeit ist nicht die natuerliche
      Ordnung der Trainingsschritte
    """
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'pta_analysis.png')
    plt.savefig(output_path, dpi=150)
    print(f"  Gespeichert: {output_path}")
    plt.close()


def run_analysis(sample_every=100, max_steps=None):
    """
    Fuehrt die vollstaendige PTA-Analyse durch.
    """
    output_dir = ensure_output_dir()
    print("=" * 70)
    print("PERMUTATION TOPOLOGY ANALYSIS (PTA)")
    print("=" * 70)

    vocab_data = load_vocabulary()
    config = vocab_data['config']
    steps_per_epoch = config['num_pairs']

    with StepReader(STEPS_FILE) as reader:
        # 1. Features extrahieren
        features, sampled_indices = build_feature_matrix(
            reader, sample_every, max_steps
        )
        print(f"\n  Feature-Matrix: {features.shape}")

        # 2. Greedy-Pfad finden
        print("\n  Finde Greedy-Pfad...")
        path = find_greedy_path(features)

        # 3. Permutations-Distanz berechnen
        metrics = compute_permutation_distance(path)
        print(f"\n  Ordnungs-Metriken:")
        print(f"    Kendall Tau: {metrics['kendall']:.3f}")
        print(f"    Spearman Rho: {metrics['spearman']:.3f}")
        print(f"    Inversions: {metrics['inversion_ratio']*100:.1f}%")

        # 4. Archetypische Cluster
        print("\n  Finde archetypische Cluster...")
        labels, centers = find_archetypal_clusters(features, n_clusters=10)

        # 5. Cluster-Timing analysieren
        cluster_timing = analyze_cluster_timing(labels, sampled_indices, steps_per_epoch)

        # Cluster mit groesstem zeitlichem Span
        max_span = max(cluster_timing.values(), key=lambda x: x['span'])
        print(f"\n  Cluster mit groesstem Span:")
        print(f"    Span: {max_span['span']} Epochen")
        print(f"    Mitglieder: {max_span['count']}")

        # 6. Visualisierungen
        print("\n  Erstelle Visualisierungen...")
        visualize_results(features, path, metrics, labels, cluster_timing,
                          sampled_indices, output_dir)

    # Ergebnisse
    results = {
        'num_samples': len(features),
        'metrics': metrics,
        'cluster_timing': cluster_timing,
        'interpretation': 'high_correlation' if metrics['kendall'] > 0.5 else 'low_correlation'
    }

    print("\n" + "=" * 70)
    print("PTA ANALYSE ABGESCHLOSSEN")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_analysis()
