"""
VOLLSTAENDIGES LOGGING EXPERIMENT
Zeichnet JEDEN der 1.050.000 Trainingsschritte auf.

Speichert:
- Jeder Schritt: Target, Context, Loss, Gradient, Aenderung
- Alle Embeddings nach jedem Schritt
- 50 komplette Durchlaeufe

Ausgabe nach: F:/!!WICHTIG/ki-forschung/

Autor: Maximilian (MI-Forschung)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import json
import os
from datetime import datetime
import struct

# ============================================================
# AUSGABE-ORDNER (5TB Speicher!)
# ============================================================

OUTPUT_BASE = "F:/!!WICHTIG/ki-forschung"
DETAIL_DIR = f"{OUTPUT_BASE}/detailed_analysis"
STEPS_DIR = f"{DETAIL_DIR}/steps"
EMBEDDINGS_DIR = f"{DETAIL_DIR}/embeddings"

# Ordner erstellen
for d in [OUTPUT_BASE, DETAIL_DIR, STEPS_DIR, EMBEDDINGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# KONFIGURATION
# ============================================================

EMBEDDING_DIM = 10
WINDOW_SIZE = 2
MIN_WORD_COUNT = 2
LEARNING_RATE = 0.01
EPOCHS = 50
NUM_RUNS = 50

# ============================================================
# TRAININGSTEXT
# ============================================================

TRAINING_TEXT = """
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

# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def tokenize(text):
    text = text.lower()
    for char in '.,!?;:()[]{}"\'-':
        text = text.replace(char, ' ')
    return text.split()

def build_vocabulary(words, min_count=2):
    word_counts = Counter(words)
    vocabulary = {}
    idx = 0
    for word, count in word_counts.items():
        if count >= min_count:
            vocabulary[word] = idx
            idx += 1
    return vocabulary, word_counts

def create_training_pairs(words, vocabulary, window_size=2):
    pairs = []
    for i, word in enumerate(words):
        if word not in vocabulary:
            continue
        start = max(0, i - window_size)
        end = min(len(words), i + window_size + 1)
        for j in range(start, end):
            if i != j and words[j] in vocabulary:
                pairs.append((vocabulary[word], vocabulary[words[j]]))
    return pairs

# ============================================================
# MODELL
# ============================================================

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, target_word):
        embedded = self.embeddings(target_word)
        return self.output(embedded)

    def get_all_embeddings(self):
        with torch.no_grad():
            return self.embeddings.weight.clone().numpy()

# ============================================================
# BINAER-WRITER FUER SCHNELLES SPEICHERN
# ============================================================

class StepLogger:
    """
    Speichert Trainingsschritte effizient im Binaerformat.

    Format pro Schritt:
    - run_id: uint16 (2 Bytes)
    - epoch: uint16 (2 Bytes)
    - step_in_epoch: uint16 (2 Bytes)
    - target_idx: uint16 (2 Bytes)
    - context_idx: uint16 (2 Bytes)
    - loss: float32 (4 Bytes)
    - gradient: 10 x float32 (40 Bytes)
    - change: 10 x float32 (40 Bytes)
    - change_magnitude: float32 (4 Bytes)
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.file = open(filepath, 'wb')
        self.step_count = 0
        # Header schreiben
        self.file.write(b'STEPLOG1')  # Magic + Version
        self.file.write(struct.pack('I', EMBEDDING_DIM))

    def log_step(self, run_id, epoch, step_in_epoch, target_idx, context_idx,
                 loss, gradient, change, change_magnitude):
        data = struct.pack(
            'HHHHHf' + 'f'*EMBEDDING_DIM + 'f'*EMBEDDING_DIM + 'f',
            run_id, epoch, step_in_epoch, target_idx, context_idx,
            loss,
            *gradient,
            *change,
            change_magnitude
        )
        self.file.write(data)
        self.step_count += 1

    def close(self):
        self.file.close()
        size_mb = os.path.getsize(self.filepath) / 1024 / 1024
        print(f"  {self.step_count:,} Schritte gespeichert ({size_mb:.1f} MB)")

class EmbeddingLogger:
    """
    Speichert Embedding-Snapshots effizient.
    """

    def __init__(self, filepath, vocab_size, embedding_dim):
        self.filepath = filepath
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.file = open(filepath, 'wb')
        self.snapshot_count = 0
        # Header
        self.file.write(b'EMBLOG01')
        self.file.write(struct.pack('II', vocab_size, embedding_dim))

    def log_embeddings(self, embeddings):
        self.file.write(embeddings.astype(np.float32).tobytes())
        self.snapshot_count += 1

    def close(self):
        self.file.close()
        size_mb = os.path.getsize(self.filepath) / 1024 / 1024
        size_gb = size_mb / 1024
        if size_gb > 1:
            print(f"  {self.snapshot_count:,} Snapshots gespeichert ({size_gb:.2f} GB)")
        else:
            print(f"  {self.snapshot_count:,} Snapshots gespeichert ({size_mb:.1f} MB)")

# ============================================================
# TRAINING MIT VOLLSTAENDIGEM LOGGING
# ============================================================

def train_with_full_logging(training_pairs, vocab_size, idx_to_word, run_id,
                           step_logger, embedding_logger):
    model = SkipGramModel(vocab_size, EMBEDDING_DIM)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Initiale Embeddings loggen
    embedding_logger.log_embeddings(model.get_all_embeddings())

    epoch_losses = []

    for epoch in range(EPOCHS):
        epoch_loss = 0
        indices = np.random.permutation(len(training_pairs))

        for step_in_epoch, idx in enumerate(indices):
            target_idx, context_idx = training_pairs[idx]

            # Embedding VOR Update
            emb_before = model.embeddings.weight[target_idx].detach().clone().numpy()

            # Forward
            optimizer.zero_grad()
            target_tensor = torch.tensor([target_idx])
            context_tensor = torch.tensor([context_idx])
            output = model(target_tensor)
            loss = criterion(output, context_tensor)

            # Backward
            loss.backward()

            # Gradient speichern
            gradient = model.embeddings.weight.grad[target_idx].clone().numpy()

            # Update
            optimizer.step()

            # Embedding NACH Update
            emb_after = model.embeddings.weight[target_idx].detach().clone().numpy()
            change = emb_after - emb_before
            change_magnitude = np.linalg.norm(change)

            # LOGGING
            step_logger.log_step(
                run_id=run_id,
                epoch=epoch,
                step_in_epoch=step_in_epoch,
                target_idx=target_idx,
                context_idx=context_idx,
                loss=loss.item(),
                gradient=gradient,
                change=change,
                change_magnitude=change_magnitude
            )

            embedding_logger.log_embeddings(model.get_all_embeddings())

            epoch_loss += loss.item()

        epoch_losses.append(epoch_loss / len(training_pairs))

    return model.get_all_embeddings(), epoch_losses

# ============================================================
# HAUPTPROGRAMM
# ============================================================

def main():
    print("=" * 70)
    print("VOLLSTAENDIGES LOGGING EXPERIMENT")
    print("=" * 70)
    print(f"Ausgabe nach: {OUTPUT_BASE}")
    print()

    start_time = datetime.now()

    # Text vorbereiten
    words = tokenize(TRAINING_TEXT)
    vocabulary, word_counts = build_vocabulary(words, MIN_WORD_COUNT)
    training_pairs = create_training_pairs(words, vocabulary, WINDOW_SIZE)
    idx_to_word = {v: k for k, v in vocabulary.items()}

    vocab_size = len(vocabulary)
    num_pairs = len(training_pairs)
    steps_per_run = num_pairs * EPOCHS
    total_steps = steps_per_run * NUM_RUNS

    print(f"Vokabular: {vocab_size} Woerter")
    print(f"Trainingspaare: {num_pairs}")
    print(f"Schritte pro Run: {steps_per_run:,}")
    print(f"Gesamtschritte: {total_steps:,}")
    print()

    # Speicherabschaetzung
    step_size = 98  # Bytes pro Schritt
    emb_size = vocab_size * EMBEDDING_DIM * 4
    total_snapshots = total_steps + NUM_RUNS

    print("Geschaetzte Speichernutzung:")
    print(f"  Steps-Log: {total_steps * step_size / 1024 / 1024:.1f} MB")
    print(f"  Embeddings: {total_snapshots * emb_size / 1024 / 1024 / 1024:.2f} GB")
    print()

    # Vokabular speichern
    vocab_path = f"{DETAIL_DIR}/vocabulary.json"
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump({
            'vocabulary': vocabulary,
            'idx_to_word': idx_to_word,
            'word_counts': dict(word_counts),
            'config': {
                'embedding_dim': EMBEDDING_DIM,
                'epochs': EPOCHS,
                'num_runs': NUM_RUNS,
                'learning_rate': LEARNING_RATE,
                'num_pairs': num_pairs
            }
        }, f, indent=2, ensure_ascii=False)
    print(f"Vokabular gespeichert: {vocab_path}")

    # Logger initialisieren
    step_logger = StepLogger(f"{STEPS_DIR}/all_steps.bin")
    embedding_logger = EmbeddingLogger(
        f"{EMBEDDINGS_DIR}/all_embeddings.bin",
        vocab_size, EMBEDDING_DIM
    )

    print()
    print("-" * 70)
    print("TRAINING STARTET")
    print("-" * 70)

    all_final_embeddings = []
    all_losses = []

    for run in range(NUM_RUNS):
        run_start = datetime.now()

        final_emb, losses = train_with_full_logging(
            training_pairs, vocab_size, idx_to_word, run,
            step_logger, embedding_logger
        )

        all_final_embeddings.append(final_emb)
        all_losses.append(losses)

        run_time = (datetime.now() - run_start).total_seconds()
        total_elapsed = (datetime.now() - start_time).total_seconds()
        eta = (total_elapsed / (run + 1)) * (NUM_RUNS - run - 1)

        print(f"Run {run+1:2d}/{NUM_RUNS} | "
              f"Loss: {losses[-1]:.4f} | "
              f"Zeit: {run_time:.1f}s | "
              f"ETA: {eta/60:.1f}min")

    print()
    print("Schliesse Log-Dateien...")
    step_logger.close()
    embedding_logger.close()

    total_time = (datetime.now() - start_time).total_seconds()

    # ============================================================
    # AUSWERTUNG
    # ============================================================

    print()
    print("=" * 70)
    print("AUSWERTUNG")
    print("=" * 70)

    final_emb_array = np.array(all_final_embeddings)
    all_losses_array = np.array(all_losses)

    print("\n--- EMBEDDING-VARIANZ PRO WORT ---")
    word_variances = {}
    for word, idx in vocabulary.items():
        word_embs = final_emb_array[:, idx, :]
        std = word_embs.std(axis=0).mean()
        word_variances[word] = float(std)

    sorted_by_var = sorted(word_variances.items(), key=lambda x: x[1], reverse=True)
    for word, var in sorted_by_var[:10]:
        freq = word_counts[word]
        print(f"  {word:12s} Varianz: {var:.4f}  (Haeufigkeit: {freq})")

    print("\n--- LOSS-STATISTIK ---")
    final_losses = all_losses_array[:, -1]
    print(f"  Mittelwert: {final_losses.mean():.4f}")
    print(f"  Std:        {final_losses.std():.4f}")
    print(f"  Min:        {final_losses.min():.4f}")
    print(f"  Max:        {final_losses.max():.4f}")

    # Zusammenfassung speichern
    summary = {
        'total_steps': int(total_steps),
        'total_time_seconds': total_time,
        'steps_per_second': total_steps / total_time,
        'final_loss_mean': float(final_losses.mean()),
        'final_loss_std': float(final_losses.std()),
        'word_variances': word_variances,
        'files': {
            'steps': f"{STEPS_DIR}/all_steps.bin",
            'embeddings': f"{EMBEDDINGS_DIR}/all_embeddings.bin",
            'vocabulary': vocab_path
        }
    }

    summary_path = f"{DETAIL_DIR}/summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # ============================================================
    # VISUALISIERUNGEN
    # ============================================================

    print("\nErstelle Visualisierungen...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Loss-Kurven
    ax = axes[0, 0]
    for i in range(NUM_RUNS):
        ax.plot(all_losses_array[i], alpha=0.3, color='blue')
    ax.plot(all_losses_array.mean(axis=0), color='red', linewidth=2, label='Mittelwert')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Loss-Entwicklung ({NUM_RUNS} Durchlaeufe)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Finale Loss-Verteilung
    ax = axes[0, 1]
    ax.hist(final_losses, bins=15, edgecolor='black', alpha=0.7)
    ax.axvline(final_losses.mean(), color='red', linestyle='--',
               label=f'Mittel: {final_losses.mean():.4f}')
    ax.set_xlabel('Finaler Loss')
    ax.set_ylabel('Anzahl Runs')
    ax.set_title('Verteilung der finalen Loss-Werte')
    ax.legend()

    # Embedding-Varianz
    ax = axes[1, 0]
    words_sorted = [w for w, _ in sorted_by_var]
    vars_sorted = [v for _, v in sorted_by_var]
    ax.barh(range(len(words_sorted)), vars_sorted, alpha=0.7)
    ax.set_yticks(range(len(words_sorted)))
    ax.set_yticklabels(words_sorted, fontsize=8)
    ax.set_xlabel('Mittlere Standardabweichung')
    ax.set_title('Embedding-Varianz pro Wort')
    ax.invert_yaxis()

    # Katze-Embedding
    ax = axes[1, 1]
    if 'katze' in vocabulary:
        katze_idx = vocabulary['katze']
        katze_embs = final_emb_array[:, katze_idx, :]
        ax.boxplot([katze_embs[:, d] for d in range(EMBEDDING_DIM)])
        ax.set_xticklabels([f'D{d}' for d in range(EMBEDDING_DIM)])
        ax.set_ylabel('Wert')
        ax.set_title("'katze' - Werteverteilung pro Dimension")
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{DETAIL_DIR}/analysis_overview.png", dpi=150)
    print(f"  Gespeichert: {DETAIL_DIR}/analysis_overview.png")

    # Finale Embeddings als numpy
    np.save(f"{DETAIL_DIR}/final_embeddings_all_runs.npy", final_emb_array)
    print(f"  Gespeichert: {DETAIL_DIR}/final_embeddings_all_runs.npy")

    print()
    print("=" * 70)
    print("FERTIG!")
    print("=" * 70)
    print(f"""
Gesamtzeit: {total_time/60:.1f} Minuten
Schritte:   {total_steps:,}
Speed:      {total_steps/total_time:.0f} Schritte/Sekunde

Dateien in {DETAIL_DIR}:
  - all_steps.bin        (Jeder Trainingsschritt)
  - all_embeddings.bin   (Alle Embedding-Snapshots)
  - vocabulary.json      (Woerter und Konfiguration)
  - summary.json         (Zusammenfassung)
  - analysis_overview.png
  - final_embeddings_all_runs.npy
    """)

if __name__ == "__main__":
    main()
