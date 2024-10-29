import argparse
import os
import random

import muspy
from tqdm import tqdm

RESOLUTION = 12
BAR_TICKS = 4 * RESOLUTION
UPPER_DURATION = 96


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    return args


def trim_music(music, start, end, move=True):
    bias = start if move else 0

    music_new = muspy.Music(
        metadata=music.metadata,
        resolution=music.resolution,
    )

    for tempo in music.tempos:
        if tempo.time <= start:
            before_last_tempo = tempo
    music_new.tempos = [muspy.Tempo(time=0, qpm=before_last_tempo.qpm)]

    music_new.time_signatures = [muspy.TimeSignature(time=0, numerator=4, denominator=4)]
    music_new.tracks = [
        muspy.Track(name="MELODY", program=0, is_drum=False),
        muspy.Track(name="CAU", program=0, is_drum=False),
        muspy.Track(name="PIANO", program=0, is_drum=False),
    ]

    track_cnt = 0
    for track in music.tracks:
        if len(track.notes) != 0:
            track_cnt += 1

    if track_cnt == 3:
        track_idx = 0
        for track in music.tracks:
            if len(track.notes) == 0:
                continue
            for note in track.notes:
                if start <= note.time < end:
                    duration = note.duration if note.duration <= UPPER_DURATION else UPPER_DURATION
                    duration = duration if duration > 0 else 1
                    music_new[track_idx].notes.append(
                        muspy.Note(
                            time=note.time - bias,
                            pitch=note.pitch,
                            duration=duration,
                            velocity=note.velocity,
                        )
                    )
            track_idx += 1
    else:
        for track in music.tracks:
            if len(track.notes) == 0:
                continue
            track_name = track.name.upper()
            if track_name == "MELODY":
                track_idx = 0
            elif track_name == "CAU" or track_name == "BRIDGE":
                track_idx = 1
            elif track_name == "PIANO":
                track_idx = 2
            for note in track.notes:
                if start <= note.time < end:
                    duration = note.duration if note.duration <= UPPER_DURATION else UPPER_DURATION
                    duration = duration if duration > 0 else 1
                    music_new[track_idx].notes.append(
                        muspy.Note(
                            time=note.time - bias,
                            pitch=note.pitch,
                            duration=duration,
                            velocity=note.velocity,
                        )
                    )

    return music_new


def main():
    args = arg_parser()

    file_names = os.listdir(os.path.join(args.indir, "POP909"))
    os.makedirs(os.path.join(args.outdir, "muspy_data"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "midi_data"), exist_ok=True)

    cnt_beat_inconsistent = 0
    cnt_contain_blank = 0
    filtered_file_name = []
    filtered_data_name = []

    for file in tqdm(file_names):
        if not file.isdecimal():
            continue

        beat_midi_path = os.path.join(args.indir, "POP909", file, "beat_midi.txt")
        with open(beat_midi_path, "r") as f:
            lines = f.readlines()
            flag_inconsistent = False
            bar_list = []
            first_bar_time = None
            for idx, line in enumerate(lines):
                time, beat1, beat2 = line.strip().split()
                time, beat1, beat2 = float(time), int(float(beat1)), int(float(beat2))
                if beat1 == 0 and beat2 == 1:
                    flag_inconsistent = True
                    break
                if first_bar_time is None and beat2 == 1:
                    first_bar_time = time
                if beat2 == 1:
                    bar_list.append(idx)

        if flag_inconsistent:
            cnt_beat_inconsistent += 1
            continue

        filtered_file_name.append(file)

        differences = [bar_list[i + 1] - bar_list[i] for i in range(len(bar_list) - 1)]
        use_bars = [False for _ in range(len(differences) - 15)]
        for idx in range(len(differences) - 15):
            if all(differences[idx + i] == differences[idx] for i in range(16)):
                use_bars[idx] = True

        differences_sum = [0 for _ in range(len(differences))]
        differences_sum[0] = 0
        for idx in range(1, len(differences)):
            differences_sum[idx] = differences_sum[idx - 1] + differences[idx - 1]

        music = muspy.read_midi(os.path.join(args.indir, "POP909", file, file + ".mid"))
        music.adjust_resolution(RESOLUTION)
        first_bar_ticks = int(first_bar_time * music.tempos[0].qpm / 60 * music.resolution)

        for idx, use_bar in enumerate(use_bars):
            if not use_bar:
                continue
            start_time = differences_sum[idx] * RESOLUTION + first_bar_ticks
            end_time = start_time + 16 * BAR_TICKS
            music_trimed = trim_music(music, start_time, end_time, move=True)
            each_bar_notes = [0 for _ in range(16)]
            for track in music_trimed.tracks:
                for note in track.notes:
                    each_bar_notes[note.time // (4 * RESOLUTION)] += 1
            if each_bar_notes.count(0) >= 1:
                cnt_contain_blank += 1
                continue
            muspy.save_json(os.path.join(args.outdir, "muspy_data", f"{file}_{differences_sum[idx]}.json"), music_trimed)
            muspy.write_midi(os.path.join(args.outdir, "midi_data", f"{file}_{differences_sum[idx]}.mid"), music_trimed)
            filtered_data_name.append(f"{file}_{differences_sum[idx]}")

    print(f"Beat inconsistent: {cnt_beat_inconsistent}/909")
    print(f"Contain blank: {cnt_contain_blank}")

    random.seed(args.seed)
    random.shuffle(filtered_file_name)

    data_num = len(filtered_file_name)
    train_files = set(filtered_file_name[: int(data_num * 0.8)])
    valid_files = set(filtered_file_name[int(data_num * 0.8) : int(data_num * 0.9)])
    test_files = set(filtered_file_name[int(data_num * 0.9) :])
    train_data = []
    valid_data = []
    test_data = []

    for data in filtered_data_name:
        data_name = data.split("_")[0]
        if data_name in train_files:
            train_data.append(data)
        elif data_name in valid_files:
            valid_data.append(data)
        elif data_name in test_files:
            test_data.append(data)
        else:
            raise ValueError("Invalid data")

    with open(os.path.join(args.outdir, "train.txt"), "w") as f:
        f.write("\n".join(train_data))
    with open(os.path.join(args.outdir, "valid.txt"), "w") as f:
        f.write("\n".join(valid_data))
    with open(os.path.join(args.outdir, "test.txt"), "w") as f:
        f.write("\n".join(test_data))

    print(f"Train: {len(train_data)}")
    print(f"Valid: {len(valid_data)}")
    print(f"Test: {len(test_data)}")


if __name__ == "__main__":
    main()
