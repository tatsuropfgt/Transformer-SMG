import muspy

from dataset import KNOWN_DURATIONS, note_indexer, remi_indexer, remi_retriever


def to_prompt(seq, representation, src_bar=16):
    if representation == "REMI+":
        idx = (seq == remi_indexer[f"bar_{src_bar-1}"]).nonzero(as_tuple=True)[0].item()
        prompt = seq[: idx + 1]
    elif representation == "NoteTuple":
        if any(seq[:, 1] == src_bar):
            idx = (seq[:, 1] == src_bar).nonzero(as_tuple=True)[0][0].item()
            prompt = seq[:idx]
        else:
            prompt = seq
    elif representation == "PianoRoll":
        assert seq.shape[0] == 769
        idx = (src_bar - 1) * 48 + 1
        prompt = seq[:idx]

    return prompt


def cleaning_seq(seq, representation, bar_idx=16):
    if bar_idx is not None:
        if representation == "REMI+":
            seq = seq.tolist()
            new_seq = []
            j = 0
            while j < len(seq):
                if seq[j] == remi_indexer[f"bar_{bar_idx - 1}"]:
                    break
                j += 1
            for i in range(j, len(seq) - 3):
                if seq[i] == remi_indexer["end-of-song"]:
                    break
                if (
                    remi_retriever[seq[i]].split("_")[0] == "position"
                    and remi_retriever[seq[i + 1]].split("_")[0] == "track"
                    and remi_retriever[seq[i + 2]].split("_")[0] == "pitch"
                    and remi_retriever[seq[i + 3]].split("_")[0] == "duration"
                ):
                    new_seq.append((
                        bar_idx - 1,
                        int(remi_retriever[seq[i]].split("_")[1]),
                        int(remi_retriever[seq[i + 1]].split("_")[1]),
                        int(remi_retriever[seq[i + 2]].split("_")[1]),
                        int(remi_retriever[seq[i + 3]].split("_")[1]),
                    ))
        elif representation == "NoteTuple":
            new_seq = []
            for i in range(len(seq)):
                if seq[i, 1] == bar_idx and seq[i, 0] == note_indexer["meta"]["note"]:
                    new_seq.append(
                        (
                            seq[i, 1].item() - 1,
                            seq[i, 2].item() - 1,
                            seq[i, 3].item() - 1,
                            seq[i, 4].item() - 1,
                            KNOWN_DURATIONS[seq[i, 5].item() - 1],
                        )
                    )
        elif representation == "PianoRoll":
            new_seq = []
            seq = seq[1:, 1:].cpu().numpy().astype(bool)
            music_0 = muspy.from_pianoroll_representation(
                seq[:, :128], resolution=12, encode_velocity=False
            )
            music_1 = muspy.from_pianoroll_representation(
                seq[:, 128:256], resolution=12, encode_velocity=False
            )
            music_2 = muspy.from_pianoroll_representation(
                seq[:, 256:], resolution=12, encode_velocity=False
            )
            for track, music in enumerate([music_0, music_1, music_2]):
                for note in music.tracks[0].notes:
                    if note.time // 48 == bar_idx - 1:
                        new_seq.append(
                            (
                                note.time // 48,
                                note.time % 48,
                                track,
                                note.pitch,
                                note.duration,
                            )
                        )

    else:
        if representation == "REMI+":
            seq = seq.tolist()
            new_seq = []
            bar_now = 0
            for i in range(len(seq) - 3):
                if remi_retriever[seq[i]].split("_")[0] == "bar":
                    bar_now = int(remi_retriever[seq[i]].split("_")[1])
                if seq[i] == remi_indexer["end-of-song"]:
                    break
                if (
                    remi_retriever[seq[i]].split("_")[0] == "position"
                    and remi_retriever[seq[i + 1]].split("_")[0] == "track"
                    and remi_retriever[seq[i + 2]].split("_")[0] == "pitch"
                    and remi_retriever[seq[i + 3]].split("_")[0] == "duration"
                ):
                    new_seq.append(
                        (
                            bar_now,
                            int(remi_retriever[seq[i]].split("_")[1]),
                            int(remi_retriever[seq[i + 1]].split("_")[1]),
                            int(remi_retriever[seq[i + 2]].split("_")[1]),
                            int(remi_retriever[seq[i + 3]].split("_")[1]),
                        )
                    )
        elif representation == "NoteTuple":
            new_seq = []
            for i in range(len(seq)):
                if seq[i, 0] == note_indexer["meta"]["note"]:
                    new_seq.append(
                        (
                            seq[i, 1].item() - 1,
                            seq[i, 2].item() - 1,
                            seq[i, 3].item() - 1,
                            seq[i, 4].item() - 1,
                            KNOWN_DURATIONS[seq[i, 5].item() - 1],
                        )
                    )
        elif representation == "PianoRoll":
            new_seq = []
            seq = seq[1:, 1:].cpu().numpy().astype(bool)
            music_0 = muspy.from_pianoroll_representation(
                seq[:, :128], resolution=12, encode_velocity=False
            )
            music_1 = muspy.from_pianoroll_representation(
                seq[:, 128:256], resolution=12, encode_velocity=False
            )
            music_2 = muspy.from_pianoroll_representation(
                seq[:, 256:], resolution=12, encode_velocity=False
            )
            for track, music in enumerate([music_0, music_1, music_2]):
                for note in music.tracks[0].notes:
                    new_seq.append(
                        (
                            note.time // 48,
                            note.time % 48,
                            track,
                            note.pitch,
                            note.duration,
                        )
                    )
    new_seq = list(set(new_seq))
    new_seq.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))

    return new_seq


def seq_to_muspy(seq, tempo=120):
    music = muspy.Music()
    music.adjust_resolution(12)
    music.tempos.append(muspy.Tempo(time=0, qpm=tempo))
    music.tracks.append(muspy.Track())
    music.tracks.append(muspy.Track())
    music.tracks.append(muspy.Track())

    for bar, position, track, pitch, duration in seq:
        music.tracks[track].append(
            muspy.Note(
                time=bar * 48 + position,
                pitch=pitch,
                duration=duration,
                velocity=64,
            )
        )

    return music
