#!/usr/bin/env python
"""
Master Script: Run All Novel Analyses
=====================================

Fuehrt alle 17 neuartigen Analysemethoden aus und speichert die Ergebnisse.
(IRPC entfernt - kein nuetzliches Signal)

Methoden:
  Original (8):
    1. GSS  - Gradient Signature Sequences
    2. CFD  - Curvature Flow Decomposition
    3. EGA  - Echo Graph Analysis
    4. SWT  - Stagnation Wavelet Transform
    5. DAI  - Dimensional Autonomy Index
    6. DDC  - Drift Direction Census
    7. PTA  - Permutation Topology Analysis
    8. AC   - Absence Cartography

  Neu (9):
    9.  FSSA - Forbidden State Sequence Analysis
    10. TRE  - Temporal Reversal Entropy
    11. MVS  - Model Violation Spectroscopy
    12. MTE  - Metric Tensor Evolution
    13. SBD  - Structural Bifurcation Detection
    14. RFA  - Repulsion Field Analysis
    15. TPA  - Topological Persistence Analysis
    16. RHC  - Reachability Horizon Collapse
    17. TSPD - Training-Specific Phenomena Detector

Usage:
    python -m novel_analyses.run_all_analyses [--quick] [--new-only]

Optionen:
    --quick     Schneller Testlauf mit reduzierten Daten
    --new-only  Nur die 9 neuen Methoden ausfuehren

Autor: Maximilian (MI-Forschung)
"""

import sys
import json
import time
from datetime import datetime
import os

# Original-Methoden (8)
from . import gradient_signatures as gss
from . import curvature_flow as cfd
from . import echo_graph as ega
from . import stagnation_wavelet as swt
# IRPC entfernt - Kuramoto R nur 0.08-0.20, kein nuetzliches Signal
from . import dimensional_autonomy as dai
from . import drift_direction as ddc
from . import permutation_topology as pta
from . import absence_cartography as ac

# Neue Methoden (9)
from . import forbidden_state_sequences as fssa
from . import temporal_reversal as tre
from . import model_violation as mvs
from . import metric_tensor as mte
from . import structural_bifurcation as sbd
from . import repulsion_field as rfa
from . import topological_persistence as tpa
from . import reachability_horizon as rhc
from . import training_phenomena as tspd

from .common.utils import ensure_output_dir


def run_all(quick_mode=False, new_only=False):
    """
    Fuehrt alle Analysen durch.

    Args:
        quick_mode: Wenn True, werden nur Subsets analysiert (fuer Tests)
        new_only: Wenn True, nur die 9 neuen Methoden ausfuehren
    """
    output_dir = ensure_output_dir()
    start_time = time.time()

    total_methods = 9 if new_only else 17

    print("=" * 70)
    print("NOVEL ANALYSES - MASTER SCRIPT")
    print("=" * 70)
    print(f"\nStartzeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Quick Mode: {quick_mode}")
    print(f"New Only: {new_only}")
    print(f"Methoden: {total_methods}")
    print(f"Output: {output_dir}")

    # Parameter fuer Quick-Mode
    if quick_mode:
        max_steps = 50000  # Nur 50k statt 1.05M
        sample_interval = 500
    else:
        max_steps = None
        sample_interval = 100

    all_results = {}
    current_method = 0

    # ================================================================
    # ORIGINAL-METHODEN (1-8)
    # ================================================================
    if not new_only:
        # ================================================================
        # 1. GRADIENT SIGNATURE SEQUENCES (GSS)
        # ================================================================
        current_method += 1
        print("\n" + "=" * 70)
        print(f"{current_method}/{total_methods}: GRADIENT SIGNATURE SEQUENCES (GSS)")
        print("=" * 70)
        try:
            t0 = time.time()
            results_gss = gss.run_analysis(max_steps=max_steps)
            all_results['gss'] = {
                'status': 'success',
                'duration': time.time() - t0,
                'num_signatures': results_gss['num_unique_signatures'],
                'num_attractors': len(results_gss['attractors']),
                'mean_entropy': results_gss['mean_entropy']
            }
            print(f"\n  GSS abgeschlossen in {time.time() - t0:.1f}s")
        except Exception as e:
            all_results['gss'] = {'status': 'error', 'error': str(e)}
            print(f"\n  GSS Fehler: {e}")

        # ================================================================
        # 2. CURVATURE FLOW DECOMPOSITION (CFD)
        # ================================================================
        current_method += 1
        print("\n" + "=" * 70)
        print(f"{current_method}/{total_methods}: CURVATURE FLOW DECOMPOSITION (CFD)")
        print("=" * 70)
        try:
            t0 = time.time()
            words = ['katze', 'hund', 'der', 'ist', 'auf'] if quick_mode else None
            results_cfd = cfd.run_analysis(words_to_analyze=words, max_steps=max_steps)
            all_results['cfd'] = {
                'status': 'success',
                'duration': time.time() - t0,
                'num_words': results_cfd['num_words'],
                'top_geodesic': results_cfd['ranking_geodesic'][:3]
            }
            print(f"\n  CFD abgeschlossen in {time.time() - t0:.1f}s")
        except Exception as e:
            all_results['cfd'] = {'status': 'error', 'error': str(e)}
            print(f"\n  CFD Fehler: {e}")

        # ================================================================
        # 3. ECHO GRAPH ANALYSIS (EGA)
        # ================================================================
        current_method += 1
        print("\n" + "=" * 70)
        print(f"{current_method}/{total_methods}: ECHO GRAPH ANALYSIS (EGA)")
        print("=" * 70)
        try:
            t0 = time.time()
            results_ega = ega.run_analysis(
                sample_interval=sample_interval,
                threshold=1.5,
                min_gap=1000,
                max_snapshots=max_steps
            )
            all_results['ega'] = {
                'status': 'success',
                'duration': time.time() - t0,
                'num_edges': results_ega['num_edges'],
                'num_chains': results_ega['num_chains'],
                'max_echo_depth': results_ega['max_echo_depth']
            }
            print(f"\n  EGA abgeschlossen in {time.time() - t0:.1f}s")
        except Exception as e:
            all_results['ega'] = {'status': 'error', 'error': str(e)}
            print(f"\n  EGA Fehler: {e}")

        # ================================================================
        # 4. STAGNATION WAVELET TRANSFORM (SWT)
        # ================================================================
        current_method += 1
        print("\n" + "=" * 70)
        print(f"{current_method}/{total_methods}: STAGNATION WAVELET TRANSFORM (SWT)")
        print("=" * 70)
        try:
            t0 = time.time()
            results_swt = swt.run_analysis(max_steps=max_steps)
            all_results['swt'] = {
                'status': 'success',
                'duration': time.time() - t0,
                'stagnation_rate': results_swt['stagnation_rate'],
                'dominant_scales': results_swt['dominant_scales'][:3]
            }
            print(f"\n  SWT abgeschlossen in {time.time() - t0:.1f}s")
        except Exception as e:
            all_results['swt'] = {'status': 'error', 'error': str(e)}
            print(f"\n  SWT Fehler: {e}")

        # ================================================================
        # 5. DIMENSIONAL AUTONOMY INDEX (DAI)
        # ================================================================
        current_method += 1
        print("\n" + "=" * 70)
        print(f"{current_method}/{total_methods}: DIMENSIONAL AUTONOMY INDEX (DAI)")
        print("=" * 70)
        try:
            t0 = time.time()
            results_dai = dai.run_analysis(max_steps=max_steps)
            all_results['dai'] = {
                'status': 'success',
                'duration': time.time() - t0,
                'mean_autonomy': results_dai['mean_autonomy'],
                'top_leader': results_dai['top_leader'],
                'num_clusters': len(results_dai['clusters'])
            }
            print(f"\n  DAI abgeschlossen in {time.time() - t0:.1f}s")
        except Exception as e:
            all_results['dai'] = {'status': 'error', 'error': str(e)}
            print(f"\n  DAI Fehler: {e}")

        # ================================================================
        # 6. DRIFT DIRECTION CENSUS (DDC)
        # ================================================================
        current_method += 1
        print("\n" + "=" * 70)
        print(f"{current_method}/{total_methods}: DRIFT DIRECTION CENSUS (DDC)")
        print("=" * 70)
        try:
            t0 = time.time()
            results_ddc = ddc.run_analysis(max_steps=max_steps)
            all_results['ddc'] = {
                'status': 'success',
                'duration': time.time() - t0,
                'global_entropy': results_ddc['global_entropy'],
                'pca_variance': results_ddc['pca_variance_explained']
            }
            print(f"\n  DDC abgeschlossen in {time.time() - t0:.1f}s")
        except Exception as e:
            all_results['ddc'] = {'status': 'error', 'error': str(e)}
            print(f"\n  DDC Fehler: {e}")

        # ================================================================
        # 7. PERMUTATION TOPOLOGY ANALYSIS (PTA)
        # ================================================================
        current_method += 1
        print("\n" + "=" * 70)
        print(f"{current_method}/{total_methods}: PERMUTATION TOPOLOGY ANALYSIS (PTA)")
        print("=" * 70)
        try:
            t0 = time.time()
            results_pta = pta.run_analysis(sample_every=sample_interval, max_steps=max_steps)
            all_results['pta'] = {
                'status': 'success',
                'duration': time.time() - t0,
                'kendall_tau': results_pta['metrics']['kendall'],
                'interpretation': results_pta['interpretation']
            }
            print(f"\n  PTA abgeschlossen in {time.time() - t0:.1f}s")
        except Exception as e:
            all_results['pta'] = {'status': 'error', 'error': str(e)}
            print(f"\n  PTA Fehler: {e}")

        # ================================================================
        # 8. ABSENCE CARTOGRAPHY (AC)
        # ================================================================
        current_method += 1
        print("\n" + "=" * 70)
        print(f"{current_method}/{total_methods}: ABSENCE CARTOGRAPHY (AC)")
        print("=" * 70)
        try:
            t0 = time.time()
            results_ac = ac.run_analysis(max_steps=max_steps)
            all_results['ac'] = {
                'status': 'success',
                'duration': time.time() - t0,
                'coverage': results_ac['coverage'],
                'num_persistent_holes': len(results_ac['persistent_holes'])
            }
            print(f"\n  AC abgeschlossen in {time.time() - t0:.1f}s")
        except Exception as e:
            all_results['ac'] = {'status': 'error', 'error': str(e)}
            print(f"\n  AC Fehler: {e}")

    # ================================================================
    # NEUE METHODEN (9-17)
    # ================================================================

    # ================================================================
    # 9. FORBIDDEN STATE SEQUENCE ANALYSIS (FSSA)
    # ================================================================
    current_method += 1
    print("\n" + "=" * 70)
    print(f"{current_method}/{total_methods}: FORBIDDEN STATE SEQUENCE ANALYSIS (FSSA)")
    print("=" * 70)
    try:
        t0 = time.time()
        results_fssa = fssa.run_analysis(max_steps=max_steps, n_clusters=100)
        all_results['fssa'] = {
            'status': 'success',
            'duration': time.time() - t0,
            'n_visited_states': results_fssa['n_visited_states'],
            'forbidden_rate': results_fssa['forbidden_rate'],
            'n_closing_events': results_fssa['n_closing_events']
        }
        print(f"\n  FSSA abgeschlossen in {time.time() - t0:.1f}s")
    except Exception as e:
        all_results['fssa'] = {'status': 'error', 'error': str(e)}
        print(f"\n  FSSA Fehler: {e}")

    # ================================================================
    # 10. TEMPORAL REVERSAL ENTROPY (TRE)
    # ================================================================
    current_method += 1
    print("\n" + "=" * 70)
    print(f"{current_method}/{total_methods}: TEMPORAL REVERSAL ENTROPY (TRE)")
    print("=" * 70)
    try:
        t0 = time.time()
        window = 500 if quick_mode else 1000
        results_tre = tre.run_analysis(max_steps=max_steps, window_size=window)
        all_results['tre'] = {
            'status': 'success',
            'duration': time.time() - t0,
            'mean_entropy': results_tre['mean_entropy'],
            'max_entropy': results_tre['max_entropy'],
            'n_peaks': results_tre['n_peaks']
        }
        print(f"\n  TRE abgeschlossen in {time.time() - t0:.1f}s")
    except Exception as e:
        all_results['tre'] = {'status': 'error', 'error': str(e)}
        print(f"\n  TRE Fehler: {e}")

    # ================================================================
    # 11. MODEL VIOLATION SPECTROSCOPY (MVS)
    # ================================================================
    current_method += 1
    print("\n" + "=" * 70)
    print(f"{current_method}/{total_methods}: MODEL VIOLATION SPECTROSCOPY (MVS)")
    print("=" * 70)
    try:
        t0 = time.time()
        results_mvs = mvs.run_analysis(max_steps=max_steps)
        n_non_gaussian = results_mvs['gaussianity']['n_non_gaussian']
        gaussian_rate = (10 - n_non_gaussian) / 10  # Rate der Gaussian-Dimensionen
        all_results['mvs'] = {
            'status': 'success',
            'duration': time.time() - t0,
            'linearity_r2': results_mvs['linearity']['mean_r2'],
            'gaussianity_rate': gaussian_rate,
            'independence_mean_mi': results_mvs['independence']['mean_mi']
        }
        print(f"\n  MVS abgeschlossen in {time.time() - t0:.1f}s")
    except Exception as e:
        all_results['mvs'] = {'status': 'error', 'error': str(e)}
        print(f"\n  MVS Fehler: {e}")

    # ================================================================
    # 12. METRIC TENSOR EVOLUTION (MTE)
    # ================================================================
    current_method += 1
    print("\n" + "=" * 70)
    print(f"{current_method}/{total_methods}: METRIC TENSOR EVOLUTION (MTE)")
    print("=" * 70)
    try:
        t0 = time.time()
        window = 500 if quick_mode else 1000
        results_mte = mte.run_analysis(max_steps=max_steps, window_size=window)
        all_results['mte'] = {
            'status': 'success',
            'duration': time.time() - t0,
            'mean_anisotropy': results_mte['mean_eigenvalue_ratio'],
            'metric_drift': results_mte['distance_evolution']['mean_distance_change'],
            'n_reorganization_events': results_mte['n_transitions']
        }
        print(f"\n  MTE abgeschlossen in {time.time() - t0:.1f}s")
    except Exception as e:
        all_results['mte'] = {'status': 'error', 'error': str(e)}
        print(f"\n  MTE Fehler: {e}")

    # ================================================================
    # 13. STRUCTURAL BIFURCATION DETECTION (SBD)
    # ================================================================
    current_method += 1
    print("\n" + "=" * 70)
    print(f"{current_method}/{total_methods}: STRUCTURAL BIFURCATION DETECTION (SBD)")
    print("=" * 70)
    try:
        t0 = time.time()
        window = 500 if quick_mode else 1000
        results_sbd = sbd.run_analysis(max_steps=max_steps, window_size=window)
        all_results['sbd'] = {
            'status': 'success',
            'duration': time.time() - t0,
            'n_bifurcations': results_sbd['n_bifurcations'],
            'mean_eigenratio': results_sbd['mean_ratio'],
            'bifurcation_rate': results_sbd['n_bifurcations'] / max(results_sbd['n_windows'], 1)
        }
        print(f"\n  SBD abgeschlossen in {time.time() - t0:.1f}s")
    except Exception as e:
        all_results['sbd'] = {'status': 'error', 'error': str(e)}
        print(f"\n  SBD Fehler: {e}")

    # ================================================================
    # 14. REPULSION FIELD ANALYSIS (RFA)
    # ================================================================
    current_method += 1
    print("\n" + "=" * 70)
    print(f"{current_method}/{total_methods}: REPULSION FIELD ANALYSIS (RFA)")
    print("=" * 70)
    try:
        t0 = time.time()
        interval = 200 if quick_mode else 100
        results_rfa = rfa.run_analysis(max_steps=max_steps, n_regions=50, sample_interval=interval)
        all_results['rfa'] = {
            'status': 'success',
            'duration': time.time() - t0,
            'n_visited_regions': results_rfa['n_visited_regions'],
            'mean_dwell_time': results_rfa['mean_dwell_time'],
            'n_repulsive_regions': results_rfa['n_repulsive_regions']
        }
        print(f"\n  RFA abgeschlossen in {time.time() - t0:.1f}s")
    except Exception as e:
        all_results['rfa'] = {'status': 'error', 'error': str(e)}
        print(f"\n  RFA Fehler: {e}")

    # ================================================================
    # 15. TOPOLOGICAL PERSISTENCE ANALYSIS (TPA)
    # ================================================================
    current_method += 1
    print("\n" + "=" * 70)
    print(f"{current_method}/{total_methods}: TOPOLOGICAL PERSISTENCE ANALYSIS (TPA)")
    print("=" * 70)
    try:
        t0 = time.time()
        interval = 50 if quick_mode else 10
        results_tpa = tpa.run_analysis(max_steps=max_steps, sample_interval=interval)
        all_results['tpa'] = {
            'status': 'success',
            'duration': time.time() - t0,
            'n_words': results_tpa['n_words'],
            'mean_complexity': results_tpa['mean_tortuosity'],
            'complexity_variance': results_tpa['mean_fractal_dim']
        }
        print(f"\n  TPA abgeschlossen in {time.time() - t0:.1f}s")
    except Exception as e:
        all_results['tpa'] = {'status': 'error', 'error': str(e)}
        print(f"\n  TPA Fehler: {e}")

    # ================================================================
    # 16. REACHABILITY HORIZON COLLAPSE (RHC)
    # ================================================================
    current_method += 1
    print("\n" + "=" * 70)
    print(f"{current_method}/{total_methods}: REACHABILITY HORIZON COLLAPSE (RHC)")
    print("=" * 70)
    try:
        t0 = time.time()
        interval = 2000 if quick_mode else 1000
        results_rhc = rhc.run_analysis(max_steps=max_steps, checkpoint_interval=interval)
        all_results['rhc'] = {
            'status': 'success',
            'duration': time.time() - t0,
            'n_words_tracked': results_rhc['n_words_tracked'],
            'mean_plasticity': results_rhc['mean_plasticity'],
            'frequency_correlation': results_rhc['frequency_correlation']
        }
        print(f"\n  RHC abgeschlossen in {time.time() - t0:.1f}s")
    except Exception as e:
        all_results['rhc'] = {'status': 'error', 'error': str(e)}
        print(f"\n  RHC Fehler: {e}")

    # ================================================================
    # 17. TRAINING-SPECIFIC PHENOMENA DETECTOR (TSPD)
    # ================================================================
    current_method += 1
    print("\n" + "=" * 70)
    print(f"{current_method}/{total_methods}: TRAINING-SPECIFIC PHENOMENA DETECTOR (TSPD)")
    print("=" * 70)
    try:
        t0 = time.time()
        window = 50 if quick_mode else 100
        results_tspd = tspd.run_analysis(max_steps=max_steps, window_size=window)
        all_results['tspd'] = {
            'status': 'success',
            'duration': time.time() - t0,
            'n_echo_chambers': results_tspd['echo_chambers']['n_pairs'],
            'n_interference_events': results_tspd['interference']['n_events'],
            'n_bursts': results_tspd['bursts']['n_bursts']
        }
        print(f"\n  TSPD abgeschlossen in {time.time() - t0:.1f}s")
    except Exception as e:
        all_results['tspd'] = {'status': 'error', 'error': str(e)}
        print(f"\n  TSPD Fehler: {e}")

    # ================================================================
    # ZUSAMMENFASSUNG
    # ================================================================
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("ZUSAMMENFASSUNG")
    print("=" * 70)

    successful = sum(1 for r in all_results.values() if r.get('status') == 'success')
    print(f"\nErfolgreich: {successful}/{total_methods}")
    print(f"Gesamtzeit: {total_time:.1f}s ({total_time/60:.1f} min)")

    print("\nOriginal-Methoden (1-8):")
    for name in ['gss', 'cfd', 'ega', 'swt', 'dai', 'ddc', 'pta', 'ac']:
        if name in all_results:
            result = all_results[name]
            status = "OK" if result.get('status') == 'success' else "FEHLER"
            duration = result.get('duration', 0)
            print(f"  {name.upper()}: {status} ({duration:.1f}s)")

    print("\nNeue Methoden (9-17):")
    for name in ['fssa', 'tre', 'mvs', 'mte', 'sbd', 'rfa', 'tpa', 'rhc', 'tspd']:
        if name in all_results:
            result = all_results[name]
            status = "OK" if result.get('status') == 'success' else "FEHLER"
            duration = result.get('duration', 0)
            print(f"  {name.upper()}: {status} ({duration:.1f}s)")

    # Ergebnisse speichern
    summary_path = os.path.join(output_dir, 'analysis_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'quick_mode': quick_mode,
            'new_only': new_only,
            'total_methods': total_methods,
            'successful': successful,
            'total_duration': total_time,
            'results': all_results
        }, f, indent=2, default=str)
    print(f"\nZusammenfassung gespeichert: {summary_path}")

    print("\n" + "=" * 70)
    print("ALLE ANALYSEN ABGESCHLOSSEN")
    print("=" * 70)

    return all_results


def main():
    """Haupteinstiegspunkt."""
    quick_mode = '--quick' in sys.argv
    new_only = '--new-only' in sys.argv

    if quick_mode:
        print("QUICK MODE aktiviert - reduzierte Daten fuer schnellen Test")
    if new_only:
        print("NEW ONLY MODE aktiviert - nur neue Methoden (9-17)")

    run_all(quick_mode=quick_mode, new_only=new_only)


if __name__ == "__main__":
    main()
