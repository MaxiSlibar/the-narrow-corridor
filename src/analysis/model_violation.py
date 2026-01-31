"""
Methode 12: Model Violation Spectroscopy (MVS)
=============================================

Konzept: Absichtlich FALSCHE Modelle anwenden, Struktur durch Versagen offenlegen.
Wo brechen einfache Annahmen zusammen? Diese Bruchstellen verraten wahre Struktur.

Tests:
1. Linearitaets-Test: Lineare Regression auf Changes, Residuen messen
2. Gaussianitaets-Test: Shapiro-Wilk pro Dimension
3. Unabhaengigkeits-Test: Mutual Information zwischen Dimensionen
4. Stationaritaets-Test: Change-Point Detection

Frage beantwortet:
- Wo ist das Training nichtlinear?
- Welche Dimensionen sind gekoppelt?
- Wann aendern sich die Statistiken?

Autor: Maximilian (MI-Forschung)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mutual_info_score
import os

from .common import StepReader, STEPS_FILE, load_vocabulary
from .common.utils import ensure_output_dir


def test_linearity(changes, window_size=1000, stride=500):
    """
    Testet ob Changes linear in der Zeit sind.
    Hohe Residuen = Nichtlinearitaet.

    Returns:
        violations: Liste von {step, residual, r2} dicts
    """
    violations = []

    for start in range(0, len(changes) - window_size, stride):
        window = changes[start:start + window_size]

        # Zeitindex
        t = np.arange(len(window)).reshape(-1, 1)

        # Fit lineare Modelle pro Dimension
        residuals = []
        r2_scores = []

        for dim in range(window.shape[1]):
            y = window[:, dim]
            lr = LinearRegression()
            lr.fit(t, y)
            predicted = lr.predict(t)

            residual = np.mean((y - predicted) ** 2)
            ss_res = np.sum((y - predicted) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-10))

            residuals.append(residual)
            r2_scores.append(r2)

        mean_residual = np.mean(residuals)
        mean_r2 = np.mean(r2_scores)

        violations.append({
            'step': start + window_size // 2,
            'mean_residual': float(mean_residual),
            'mean_r2': float(mean_r2),
            'per_dim_residual': [float(r) for r in residuals]
        })

    return violations


def test_gaussianity(changes, sample_size=5000):
    """
    Testet ob Changes Gaussian-verteilt sind pro Dimension.
    Niedrige p-Werte = Non-Gaussian.

    Returns:
        results: Dict pro Dimension mit p-Wert und Statistik
    """
    results = {}

    # Sample wenn zu gross
    if len(changes) > sample_size:
        indices = np.random.choice(len(changes), sample_size, replace=False)
        sample = changes[indices]
    else:
        sample = changes

    for dim in range(changes.shape[1]):
        data = sample[:, dim]

        # Shapiro-Wilk Test (max 5000 samples)
        try:
            statistic, p_value = stats.shapiro(data[:5000])
        except Exception:
            statistic, p_value = 0, 1

        # Zusaetzliche Statistiken
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)

        results[dim] = {
            'shapiro_statistic': float(statistic),
            'shapiro_p_value': float(p_value),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'is_gaussian': p_value > 0.05
        }

    return results


def test_independence(changes, n_bins=10, sample_size=10000):
    """
    Testet ob Dimensionen unabhaengig sind.
    Hohe MI = Dimensionen sind gekoppelt.

    Returns:
        mi_matrix: Matrix der Mutual Information zwischen Dimensionen
    """
    n_dims = changes.shape[1]
    mi_matrix = np.zeros((n_dims, n_dims))

    # Sample wenn zu gross
    if len(changes) > sample_size:
        indices = np.random.choice(len(changes), sample_size, replace=False)
        sample = changes[indices]
    else:
        sample = changes

    # Diskretisiere fuer MI-Berechnung
    for i in range(n_dims):
        for j in range(i, n_dims):
            # Diskretisiere in Bins
            xi = np.digitize(sample[:, i], np.percentile(sample[:, i], np.linspace(0, 100, n_bins + 1)[1:-1]))
            xj = np.digitize(sample[:, j], np.percentile(sample[:, j], np.linspace(0, 100, n_bins + 1)[1:-1]))

            mi = mutual_info_score(xi, xj)
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi

    return mi_matrix


def test_stationarity(changes, window_size=10000, stride=5000):
    """
    Testet ob Statistiken stationaer sind (aendern sich ueber Zeit?).
    Grosse Aenderungen = Non-stationaer.

    Returns:
        change_points: Liste von {step, metric_change} dicts
    """
    change_points = []

    prev_stats = None

    for start in range(0, len(changes) - window_size, stride):
        window = changes[start:start + window_size]

        # Berechne Statistiken fuer dieses Fenster
        current_stats = {
            'mean': np.mean(window, axis=0),
            'std': np.std(window, axis=0),
            'median': np.median(window, axis=0)
        }

        if prev_stats is not None:
            # Messe Aenderung der Statistiken
            mean_change = np.linalg.norm(current_stats['mean'] - prev_stats['mean'])
            std_change = np.linalg.norm(current_stats['std'] - prev_stats['std'])

            total_change = mean_change + std_change

            change_points.append({
                'step': start + window_size // 2,
                'mean_change': float(mean_change),
                'std_change': float(std_change),
                'total_change': float(total_change)
            })

        prev_stats = current_stats

    return change_points


def find_significant_violations(linearity_results, gaussianity_results,
                                 independence_results, stationarity_results):
    """
    Identifiziert die signifikantesten Modell-Verletzungen.

    Returns:
        violations: Dict mit kategorisierten Verletzungen
    """
    violations = {
        'linearity': [],
        'gaussianity': [],
        'independence': [],
        'stationarity': []
    }

    # Linearitaet: Niedrige R2 = schlechte Anpassung
    if linearity_results:
        r2_values = [v['mean_r2'] for v in linearity_results]
        threshold = np.percentile(r2_values, 10)  # Schlechteste 10%
        violations['linearity'] = [
            v for v in linearity_results if v['mean_r2'] < threshold
        ]

    # Gaussianitaet: Niedrige p-Werte = Non-Gaussian
    non_gaussian_dims = [
        dim for dim, result in gaussianity_results.items()
        if not result['is_gaussian']
    ]
    violations['gaussianity'] = non_gaussian_dims

    # Unabhaengigkeit: Hohe MI = gekoppelt
    if independence_results is not None:
        n_dims = independence_results.shape[0]
        coupled_pairs = []
        mi_threshold = np.percentile(independence_results[independence_results > 0], 90)

        for i in range(n_dims):
            for j in range(i + 1, n_dims):
                if independence_results[i, j] > mi_threshold:
                    coupled_pairs.append((i, j, float(independence_results[i, j])))

        violations['independence'] = coupled_pairs

    # Stationaritaet: Grosse Aenderungen = Change Points
    if stationarity_results:
        changes = [cp['total_change'] for cp in stationarity_results]
        threshold = np.percentile(changes, 90)
        violations['stationarity'] = [
            cp for cp in stationarity_results if cp['total_change'] > threshold
        ]

    return violations


def visualize_results(linearity_results, gaussianity_results,
                      independence_results, stationarity_results,
                      violations, output_dir):
    """Erstellt Visualisierungen."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Linearitaet ueber Zeit
    ax = axes[0, 0]
    if linearity_results:
        steps = [v['step'] for v in linearity_results]
        r2 = [v['mean_r2'] for v in linearity_results]
        ax.plot(steps, r2, 'b-', alpha=0.7)
        ax.axhline(np.mean(r2), color='r', linestyle='--', label=f'Mean: {np.mean(r2):.3f}')
    ax.set_xlabel('Trainingsschritt')
    ax.set_ylabel('R² (Linearity)')
    ax.set_title('Linearitaets-Test (niedriger = nichtlinear)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Gaussianity pro Dimension
    ax = axes[0, 1]
    dims = list(gaussianity_results.keys())
    p_values = [gaussianity_results[d]['shapiro_p_value'] for d in dims]
    colors = ['red' if p < 0.05 else 'green' for p in p_values]
    ax.bar(dims, p_values, color=colors, alpha=0.7)
    ax.axhline(0.05, color='black', linestyle='--', label='p=0.05')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Shapiro-Wilk p-Wert')
    ax.set_title('Gaussianitaets-Test (rot = non-Gaussian)')
    ax.legend()

    # 3. Independence Matrix
    ax = axes[0, 2]
    if independence_results is not None:
        im = ax.imshow(independence_results, cmap='hot', aspect='auto')
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Dimension')
        ax.set_title('Mutual Information Matrix (hoch = gekoppelt)')
        plt.colorbar(im, ax=ax)

    # 4. Stationarity Change Points
    ax = axes[1, 0]
    if stationarity_results:
        steps = [cp['step'] for cp in stationarity_results]
        changes = [cp['total_change'] for cp in stationarity_results]
        ax.plot(steps, changes, 'b-', alpha=0.7)
        threshold = np.percentile(changes, 90)
        ax.axhline(threshold, color='r', linestyle='--', label='90th percentile')

        # Markiere Change Points
        for cp in violations['stationarity'][:10]:
            ax.axvline(cp['step'], color='orange', alpha=0.5)

    ax.set_xlabel('Trainingsschritt')
    ax.set_ylabel('Statistik-Aenderung')
    ax.set_title('Stationaritaets-Test (Peaks = Change Points)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Skewness/Kurtosis pro Dimension
    ax = axes[1, 1]
    skewness = [gaussianity_results[d]['skewness'] for d in dims]
    kurtosis = [gaussianity_results[d]['kurtosis'] for d in dims]
    x = np.arange(len(dims))
    width = 0.35
    ax.bar(x - width/2, skewness, width, label='Skewness', alpha=0.7)
    ax.bar(x + width/2, kurtosis, width, label='Kurtosis', alpha=0.7)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Wert')
    ax.set_title('Skewness & Kurtosis (0 = Gaussian)')
    ax.legend()
    ax.set_xticks(x)

    # 6. Zusammenfassung
    ax = axes[1, 2]
    ax.axis('off')

    n_non_gaussian = len(violations['gaussianity'])
    n_coupled = len(violations['independence'])
    n_change_points = len(violations['stationarity'])

    summary = f"""
    MODEL VIOLATION SPECTROSCOPY (MVS)
    ==================================

    LINEARITAET:
    - Fenster analysiert: {len(linearity_results)}
    - Mean R²: {np.mean([v['mean_r2'] for v in linearity_results]):.3f}
    - Min R² (schlechteste): {np.min([v['mean_r2'] for v in linearity_results]):.3f}
    - Verletzungen (R² < 10%): {len(violations['linearity'])}

    GAUSSIANITAET:
    - Dimensionen analysiert: {len(gaussianity_results)}
    - Non-Gaussian Dimensionen: {n_non_gaussian} / 10
    - Non-Gaussian Dims: {violations['gaussianity'][:5]}...

    UNABHAENGIGKEIT:
    - Stark gekoppelte Paare: {n_coupled}
    - Top gekoppelt: {violations['independence'][:3] if violations['independence'] else 'Keine'}

    STATIONARITAET:
    - Change Points gefunden: {n_change_points}
    - Top Change Points: {[cp['step'] for cp in violations['stationarity'][:5]]}

    INTERPRETATION:
    - Niedrige R² = Training ist nichtlinear in der Zeit
    - Non-Gaussian = Schwere Tails oder Asymmetrie
    - Hohe MI = Dimensionen bewegen sich zusammen
    - Change Points = Regime-Wechsel im Training
    """
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'mvs_analysis.png')
    plt.savefig(output_path, dpi=150)
    print(f"  Gespeichert: {output_path}")
    plt.close()


def run_analysis(max_steps=None, linearity_window=1000, stationarity_window=10000):
    """
    Fuehrt die vollstaendige MVS-Analyse durch.

    Args:
        max_steps: Maximale Schritte (None = alle)
        linearity_window: Fenstergroesse fuer Linearitaets-Test
        stationarity_window: Fenstergroesse fuer Stationaritaets-Test
    """
    output_dir = ensure_output_dir()

    print("=" * 70)
    print("MODEL VIOLATION SPECTROSCOPY (MVS)")
    print("=" * 70)

    print(f"\n  Konfiguration:")
    print(f"    Linearity Window: {linearity_window}")
    print(f"    Stationarity Window: {stationarity_window}")

    with StepReader(STEPS_FILE) as reader:
        # Lade Changes
        print("\n[1/5] Lade Change-Vektoren...")
        changes = reader.get_all_changes(max_steps)
        print(f"  {len(changes):,} Changes geladen")

        # Test 1: Linearitaet
        print("\n[2/5] Linearitaets-Test...")
        linearity_results = test_linearity(changes, linearity_window, linearity_window // 2)
        print(f"  {len(linearity_results)} Fenster getestet")
        print(f"  Mean R²: {np.mean([v['mean_r2'] for v in linearity_results]):.3f}")

        # Test 2: Gaussianity
        print("\n[3/5] Gaussianitaets-Test...")
        gaussianity_results = test_gaussianity(changes)
        n_non_gaussian = sum(1 for r in gaussianity_results.values() if not r['is_gaussian'])
        print(f"  Non-Gaussian Dimensionen: {n_non_gaussian} / {len(gaussianity_results)}")

        # Test 3: Independence
        print("\n[4/5] Unabhaengigkeits-Test...")
        independence_results = test_independence(changes)
        max_mi = np.max(independence_results[np.triu_indices(10, k=1)])
        print(f"  Max Mutual Information: {max_mi:.3f}")

        # Test 4: Stationarity
        print("\n[5/5] Stationaritaets-Test...")
        stationarity_results = test_stationarity(changes, stationarity_window, stationarity_window // 2)
        print(f"  {len(stationarity_results)} Fenster-Uebergaenge analysiert")

        # Finde signifikante Verletzungen
        print("\n  Identifiziere signifikante Verletzungen...")
        violations = find_significant_violations(
            linearity_results, gaussianity_results,
            independence_results, stationarity_results
        )

        # Visualisierungen
        print("\n  Erstelle Visualisierungen...")
        visualize_results(
            linearity_results, gaussianity_results,
            independence_results, stationarity_results,
            violations, output_dir
        )

    # Ergebnisse
    results = {
        'linearity': {
            'n_windows': len(linearity_results),
            'mean_r2': float(np.mean([v['mean_r2'] for v in linearity_results])),
            'min_r2': float(np.min([v['mean_r2'] for v in linearity_results])),
            'n_violations': len(violations['linearity'])
        },
        'gaussianity': {
            'n_non_gaussian': sum(1 for r in gaussianity_results.values() if not r['is_gaussian']),
            'per_dim': gaussianity_results
        },
        'independence': {
            'max_mi': float(np.max(independence_results[np.triu_indices(10, k=1)])),
            'mean_mi': float(np.mean(independence_results[np.triu_indices(10, k=1)])),
            'n_coupled_pairs': len(violations['independence']),
            'coupled_pairs': violations['independence'][:10]
        },
        'stationarity': {
            'n_change_points': len(violations['stationarity']),
            'change_points': [cp['step'] for cp in violations['stationarity'][:10]]
        }
    }

    print("\n" + "=" * 70)
    print("MVS ANALYSE ABGESCHLOSSEN")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_analysis()
