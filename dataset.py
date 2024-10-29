import os

import muspy
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

MAX_BAR = 16
MAX_POSITION = 48
MAX_TRACKS = 3
MAX_PITCH = 128
MAX_DURATION = 96
KNOWN_DURATIONS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    15,
    16,
    18,
    20,
    21,
    24,
    30,
    36,
    40,
    42,
    48,
    60,
    72,
    84,
    96,
]
DURATION_MAP = {
    i: KNOWN_DURATIONS[np.argmin(np.abs(np.array(KNOWN_DURATIONS) - i))]
    for i in range(1, MAX_DURATION + 1)
}
MAX_DURATION = len(KNOWN_DURATIONS)
IGNORE_INDEX = -100

remi_indexer = {}
remi_indexer["pad"] = 0
remi_indexer["start-of-song"] = 1
remi_indexer["end-of-song"] = 2
bias = 3

for i in range(MAX_BAR):
    remi_indexer[f"bar_{i}"] = i + bias
bias += MAX_BAR

for i in range(MAX_POSITION):
    remi_indexer[f"position_{i}"] = i + bias
bias += MAX_POSITION

for i in range(MAX_TRACKS):
    remi_indexer[f"track_{i}"] = i + bias
bias += MAX_TRACKS

for i in range(MAX_PITCH):
    remi_indexer[f"pitch_{i}"] = i + bias
bias += MAX_PITCH

for idx, duration in enumerate(KNOWN_DURATIONS):
    remi_indexer[f"duration_{duration}"] = idx + bias

remi_num_tokens = len(remi_indexer)
remi_retriever = {v: k for k, v in remi_indexer.items()}

note_indexer = {}
note_indexer["meta"] = {}
note_indexer["bar"] = {}
note_indexer["position"] = {}
note_indexer["tracks"] = {}
note_indexer["pitch"] = {}
note_indexer["duration"] = {}

note_indexer["meta"]["pad"] = 0
note_indexer["meta"]["start-of-song"] = 1
note_indexer["meta"]["note"] = 2
note_indexer["meta"]["end-of-song"] = 3

for i in range(MAX_BAR):
    note_indexer["bar"][i] = i + 1

for i in range(MAX_POSITION):
    note_indexer["position"][i] = i + 1

for i in range(MAX_TRACKS):
    note_indexer["tracks"][i] = i + 1

for i in range(MAX_PITCH):
    note_indexer["pitch"][i] = i + 1

for idx, duration in enumerate(KNOWN_DURATIONS):
    note_indexer["duration"][duration] = idx + 1

note_event_order = ["meta", "bar", "position", "tracks", "pitch", "duration"]
note_num_tokens = {
    "meta": 4,
    "bar": MAX_BAR + 1,
    "position": MAX_POSITION + 1,
    "tracks": MAX_TRACKS + 1,
    "pitch": MAX_PITCH + 1,
    "duration": MAX_DURATION + 1,
}

proll_num_token = MAX_PITCH * MAX_TRACKS + 1
proll_seq_len = MAX_BAR * MAX_POSITION


def get_dataloaders(cfg, music_rep):
    dataset_classes = {
        "REMI+": REMIPlusDataset,
        "NoteTuple": NoteTupleDataset,
        "PianoRoll": PianoRollDataset,
    }
    train_dataset = dataset_classes[music_rep](
        cfg.data_root, cfg.data_src, cfg.train_file, cfg.transpose
    )
    val_dataset = dataset_classes[music_rep](
        cfg.data_root, cfg.data_src, cfg.val_file, False
    )
    test_dataset = dataset_classes[music_rep](
        cfg.data_root, cfg.data_src, cfg.test_file, False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=True, num_workers=cfg.num_workers
    )

    return train_loader, val_loader, test_loader


class MusicDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        data_src: str,
        data_file: str,
        transpose: bool = False,
        max_note_len: int = 1024,
        # output_tempo: bool = False,
    ):
        super().__init__()
        self.data_dir = os.path.join(data_root, data_src)
        self.data_file = os.path.join(data_root, data_file)

        self.transpose = transpose
        self.max_note_len = max_note_len

        with open(self.data_file, "r") as f:
            self.dataset = f.read().splitlines()

        self.data_sum = len(self.dataset)
        self.output_tempo = False

    def __getitem__(self, index):
        file_name = os.path.join(self.data_dir, self.dataset[index])
        music = muspy.load(file_name + ".json")
        if self.transpose:
            music.transpose(np.random.randint(-6, 7))
        seq = self.music_to_representation(music)
        if self.output_tempo:
            tempo = music.tempos[0].qpm
            return seq, tempo
        return seq

    def __len__(self):
        return self.data_sum


class REMIPlusDataset(MusicDataset):
    """REMI+ representation

    paper: https://arxiv.org/abs/2201.10936
    implementation ref: https://github.dev/salu133445/mmt/blob/main/baseline/representation_remi.py
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_len = self.max_note_len * 4

    def music_to_representation(self, music):
        resolution = music.resolution
        notes = []
        for track_idx, track in enumerate(music.tracks):
            for note in track.notes:
                instrument = track_idx
                bar = note.time // (resolution * 4)
                position = note.time % (resolution * 4)
                pitch = note.pitch
                duration = note.duration
                notes.append([instrument, bar, position, pitch, duration])
        notes.sort(key=lambda x: (x[1], x[2], x[3], x[4], x[0]))

        seq = torch.full((self.seq_len,), IGNORE_INDEX, dtype=torch.int16)
        seq[0] = remi_indexer["start-of-song"]
        last_bar = -1
        idx = 1
        for instrument, bar, position, pitch, duration in notes:
            if bar > last_bar:
                last_bar = bar
                seq[idx] = remi_indexer[f"bar_{bar}"]
                idx += 1
            seq[idx] = remi_indexer[f"position_{position}"]
            seq[idx + 1] = remi_indexer[f"track_{instrument}"]
            seq[idx + 2] = remi_indexer[f"pitch_{pitch}"]
            seq[idx + 3] = remi_indexer[
                f"duration_{DURATION_MAP[min(duration, MAX_DURATION)]}"
            ]
            idx += 4
        seq[idx] = remi_indexer["end-of-song"]
        return seq.long()


class NoteTupleDataset(MusicDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_len = self.max_note_len

    def music_to_representation(self, music):
        resolution = music.resolution
        notes = []
        for track_idx, track in enumerate(music.tracks):
            for note in track.notes:
                bar = note.time // (resolution * 4)
                position = note.time % (resolution * 4)
                pitch = note.pitch
                duration = note.duration
                notes.append([bar, position, track_idx, pitch, duration])
        notes.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))

        seq = torch.full((self.seq_len, 6), IGNORE_INDEX, dtype=torch.int16)
        seq[0] = torch.tensor(
            [
                note_indexer["meta"]["start-of-song"],
                IGNORE_INDEX,
                IGNORE_INDEX,
                IGNORE_INDEX,
                IGNORE_INDEX,
                IGNORE_INDEX,
            ]
        )
        idx = 1
        for bar, position, track, pitch, duration in notes:
            seq[idx] = torch.tensor(
                [
                    note_indexer["meta"]["note"],
                    note_indexer["bar"][bar],
                    note_indexer["position"][position],
                    note_indexer["tracks"][track],
                    note_indexer["pitch"][pitch],
                    note_indexer["duration"][DURATION_MAP[min(duration, MAX_DURATION)]],
                ]
            )
            idx += 1

        seq[idx] = torch.tensor(
            [
                note_indexer["meta"]["end-of-song"],
                IGNORE_INDEX,
                IGNORE_INDEX,
                IGNORE_INDEX,
                IGNORE_INDEX,
                IGNORE_INDEX,
            ]
        )
        return seq.long()


class PianoRollDataset(MusicDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_len = proll_seq_len
        self.num_tokens = proll_num_token
        self.max_pitch = MAX_PITCH

    def music_to_representation(self, music):
        seq_len = self.seq_len
        piano_roll = np.zeros((seq_len, self.num_tokens - 1), dtype=bool)
        for track_idx, track in enumerate(music.tracks):
            if track.notes == []:
                continue
            new_music = muspy.Music(resolution=music.resolution, tracks=[track])
            pr = muspy.to_pianoroll_representation(new_music, encode_velocity=False)
            if pr.shape[0] > seq_len:
                pr = pr[:seq_len]
            elif pr.shape[0] < seq_len:
                pr = np.pad(
                    pr, ((0, seq_len - pr.shape[0]), (0, 0)), constant_values=False
                )
            piano_roll[
                :, self.max_pitch * track_idx : self.max_pitch * (track_idx + 1)
            ] = pr
        # add cls token
        piano_roll = np.pad(piano_roll, ((1, 0), (1, 0)), constant_values=False)
        piano_roll[0, 0] = True
        piano_roll = torch.tensor(piano_roll, dtype=torch.float32)  # (769, 128 * 3 + 1)
        return piano_roll
