"""
Common Utilities
================

Gemeinsame Pfade, Konfiguration und Hilfsfunktionen.
"""

import json
import os
import numpy as np

# Pfade
DATA_DIR = "F:/!!WICHTIG/ki-forschung/detailed_analysis"
STEPS_FILE = f"{DATA_DIR}/steps/all_steps.bin"
EMBEDDINGS_FILE = f"{DATA_DIR}/embeddings/all_embeddings.bin"
VOCAB_FILE = f"{DATA_DIR}/vocabulary.json"
OUTPUT_DIR = "F:/!!WICHTIG/ki-forschung/novel_analysis_results"


def ensure_output_dir():
    """Erstellt Output-Verzeichnis falls nicht vorhanden."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def load_vocabulary():
    """Laedt Vokabular und Konfiguration."""
    with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {
        'vocabulary': data['vocabulary'],          # word -> idx
        'idx_to_word': data['idx_to_word'],        # idx -> word
        'word_counts': data.get('word_counts', {}),
        'config': data['config']
    }


def get_config():
    """Laedt nur die Konfiguration."""
    vocab_data = load_vocabulary()
    return vocab_data['config']


def word_to_idx(word, vocabulary):
    """Wort zu Index."""
    return vocabulary.get(word.lower())


def idx_to_word(idx, idx_to_word_map):
    """Index zu Wort."""
    return idx_to_word_map.get(str(idx))


def steps_per_run(config):
    """Berechnet Anzahl Schritte pro Run."""
    return config['num_pairs'] * config['epochs']


def format_number(n):
    """Formatiert grosse Zahlen mit Tausendertrennzeichen."""
    return f"{n:,}"


def normalize(arr, axis=None):
    """Normalisiert Array auf Einheitslaenge."""
    norm = np.linalg.norm(arr, axis=axis, keepdims=True)
    return arr / (norm + 1e-10)


def safe_divide(a, b, default=0.0):
    """Sichere Division, vermeidet Division durch Null."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(b != 0, a / b, default)
    return result
