"""
Binary Log Readers
==================

Klassen zum Lesen der binaeren Log-Dateien aus dem Training.
Basiert auf analyse_detailed_logs.py, erweitert fuer Novel Analyses.
"""

import numpy as np
import struct
import os


class StepReader:
    """Liest die Step-Log Datei (all_steps.bin)."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.file = open(filepath, 'rb')

        # Header lesen
        magic = self.file.read(8)
        if magic != b'STEPLOG1':
            raise ValueError(f"Ungueltige Datei: {magic}")

        self.embedding_dim = struct.unpack('I', self.file.read(4))[0]
        self.header_size = 12

        # Schrittgroesse mit struct.calcsize berechnen
        self.fmt = 'HHHHHf' + 'f'*self.embedding_dim + 'f'*self.embedding_dim + 'f'
        self.step_size = struct.calcsize(self.fmt)

        # Anzahl Schritte
        file_size = os.path.getsize(filepath)
        self.num_steps = (file_size - self.header_size) // self.step_size

        print(f"StepReader: {self.num_steps:,} Schritte, {self.embedding_dim} Dimensionen")

    def read_step(self, step_idx):
        """Liest einen einzelnen Schritt."""
        self.file.seek(self.header_size + step_idx * self.step_size)
        data = self.file.read(self.step_size)
        values = struct.unpack(self.fmt, data)

        return {
            'run_id': values[0],
            'epoch': values[1],
            'step_in_epoch': values[2],
            'target_idx': values[3],
            'context_idx': values[4],
            'loss': values[5],
            'gradient': np.array(values[6:6+self.embedding_dim]),
            'change': np.array(values[6+self.embedding_dim:6+2*self.embedding_dim]),
            'change_magnitude': values[-1]
        }

    def read_all_steps(self, max_steps=None, progress=True):
        """Liest alle Schritte (oder max_steps)."""
        n = min(self.num_steps, max_steps) if max_steps else self.num_steps
        steps = []

        for i in range(n):
            steps.append(self.read_step(i))
            if progress and (i + 1) % 100000 == 0:
                print(f"  Gelesen: {i+1:,} / {n:,}")

        return steps

    def read_steps_for_run(self, run_id):
        """Alle Schritte eines bestimmten Runs."""
        # Runs sind sequentiell gespeichert
        steps_per_run = self.num_steps // 50  # 50 Runs
        start_idx = run_id * steps_per_run
        end_idx = start_idx + steps_per_run

        steps = []
        for i in range(start_idx, min(end_idx, self.num_steps)):
            step = self.read_step(i)
            if step['run_id'] == run_id:
                steps.append(step)
        return steps

    def iter_steps(self, start=0, end=None, step=1):
        """Generator fuer Schritte (speichereffizient)."""
        if end is None:
            end = self.num_steps
        for i in range(start, end, step):
            yield self.read_step(i)

    def get_all_gradients(self, max_steps=None):
        """Extrahiere alle Gradienten als numpy Array."""
        n = min(self.num_steps, max_steps) if max_steps else self.num_steps
        gradients = np.zeros((n, self.embedding_dim), dtype=np.float32)

        for i in range(n):
            step = self.read_step(i)
            gradients[i] = step['gradient']
            if (i + 1) % 100000 == 0:
                print(f"  Gradienten: {i+1:,} / {n:,}")

        return gradients

    def get_all_changes(self, max_steps=None):
        """Extrahiere alle Changes als numpy Array."""
        n = min(self.num_steps, max_steps) if max_steps else self.num_steps
        changes = np.zeros((n, self.embedding_dim), dtype=np.float32)

        for i in range(n):
            step = self.read_step(i)
            changes[i] = step['change']
            if (i + 1) % 100000 == 0:
                print(f"  Changes: {i+1:,} / {n:,}")

        return changes

    def get_all_change_magnitudes(self, max_steps=None):
        """Extrahiere alle Change-Magnitudes als numpy Array."""
        n = min(self.num_steps, max_steps) if max_steps else self.num_steps
        magnitudes = np.zeros(n, dtype=np.float32)

        for i in range(n):
            step = self.read_step(i)
            magnitudes[i] = step['change_magnitude']

        return magnitudes

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class EmbeddingReader:
    """Liest die Embedding-Snapshots (all_embeddings.bin)."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.file = open(filepath, 'rb')

        # Header lesen
        magic = self.file.read(8)
        if magic != b'EMBLOG01':
            raise ValueError(f"Ungueltige Datei: {magic}")

        self.vocab_size, self.embedding_dim = struct.unpack('II', self.file.read(8))
        self.header_size = 16

        # Snapshot-Groesse
        self.snapshot_size = self.vocab_size * self.embedding_dim * 4

        # Anzahl Snapshots
        file_size = os.path.getsize(filepath)
        self.num_snapshots = (file_size - self.header_size) // self.snapshot_size

        print(f"EmbeddingReader: {self.num_snapshots:,} Snapshots, "
              f"{self.vocab_size} Woerter, {self.embedding_dim} Dimensionen")

    def read_snapshot(self, snapshot_idx):
        """Liest einen Snapshot."""
        self.file.seek(self.header_size + snapshot_idx * self.snapshot_size)
        data = self.file.read(self.snapshot_size)
        return np.frombuffer(data, dtype=np.float32).reshape(
            self.vocab_size, self.embedding_dim
        ).copy()

    def read_word_trajectory(self, word_idx, start=0, end=None, step=1):
        """Liest die komplette Trajektorie eines Wortes."""
        if end is None:
            end = self.num_snapshots

        trajectory = []
        for i in range(start, end, step):
            snapshot = self.read_snapshot(i)
            trajectory.append(snapshot[word_idx].copy())

        return np.array(trajectory)

    def read_all_trajectories(self, start=0, end=None, step=1):
        """Liest Trajektorien aller Woerter."""
        if end is None:
            end = self.num_snapshots

        n_samples = (end - start + step - 1) // step
        trajectories = np.zeros(
            (n_samples, self.vocab_size, self.embedding_dim),
            dtype=np.float32
        )

        for idx, i in enumerate(range(start, end, step)):
            trajectories[idx] = self.read_snapshot(i)
            if (idx + 1) % 1000 == 0:
                print(f"  Snapshots: {idx+1:,} / {n_samples:,}")

        return trajectories

    def iter_snapshots(self, start=0, end=None, step=1):
        """Generator fuer Snapshots (speichereffizient)."""
        if end is None:
            end = self.num_snapshots
        for i in range(start, end, step):
            yield i, self.read_snapshot(i)

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
