import numpy as np


def get_obj_metrics_all(gen_seq_all, tgt_seq_all):
    note_f1_all = note_f1(gen_seq_all, tgt_seq_all)
    pianoroll_f1_all = pianoroll_f1(gen_seq_all, tgt_seq_all)
    pitch_range_similarity_all = pitch_range_similarity(gen_seq_all, tgt_seq_all)
    note_density_similarity_all = note_density_similarity(gen_seq_all, tgt_seq_all)
    chroma_similarity_all = chroma_similarity(gen_seq_all, tgt_seq_all)
    grooving_similarity_all = grooving_similarity(gen_seq_all, tgt_seq_all)
    pitch_class_entropy_all = pitch_class_entropy(gen_seq_all)
    return {
        "note_f1": np.mean(note_f1_all),
        "pianoroll_f1": np.mean(pianoroll_f1_all),
        "pitch_range_similarity": np.mean(pitch_range_similarity_all),
        "note_density_similarity": np.mean(note_density_similarity_all),
        "chroma_similarity": np.mean(chroma_similarity_all),
        "grooving_similarity": np.mean(grooving_similarity_all),
        "pitch_class_entropy": np.mean(pitch_class_entropy_all),
    }


def note_f1(x, y):
    """
    TP if bar, position, track, pitch are all the same (duration can be different)
    """
    note_f1 = []
    for i in range(len(x)):
        x_np = np.array(x[i])
        y_np = np.array(y[i])
        if x_np.size == 0 or y_np.size == 0:
            note_f1.append(0)
            continue
        x_set = set(map(tuple, x_np[:, :4]))
        y_set = set(map(tuple, y_np[:, :4]))
        TP = len(x_set & y_set)
        FP = len(x_set - y_set)
        FN = len(y_set - x_set)
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        note_f1.append(f1)
    return note_f1


def pianoroll_f1(x, y):
    pianoroll_f1 = []
    for i in range(len(x)):
        x_pianoroll = np.zeros((128, 48), dtype=np.int32)
        y_pianoroll = np.zeros((128, 48), dtype=np.int32)
        for bar, position, track, pitch, duration in x[i]:
            x_pianoroll[pitch, position : position + duration] = 1
        for bar, position, track, pitch, duration in y[i]:
            y_pianoroll[pitch, position : position + duration] = 1
        TP = np.sum(x_pianoroll & y_pianoroll)
        FP = np.sum(x_pianoroll & ~y_pianoroll)
        FN = np.sum(~x_pianoroll & y_pianoroll)
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        pianoroll_f1.append(f1)
    return pianoroll_f1


def pitch_range_similarity(x, y):
    pr_sim = []
    for i in range(len(x)):
        x_np = np.array(x[i])
        y_np = np.array(y[i])
        if x_np.size == 0 or y_np.size == 0:
            pr_sim.append(0)
            continue
        x_pr = np.max(x_np[:, 3]) - np.min(x_np[:, 3])
        y_pr = np.max(y_np[:, 3]) - np.min(y_np[:, 3])
        pr_sim_tmp = 1 - abs(x_pr - y_pr) / 127
        pr_sim.append(pr_sim_tmp)
    return pr_sim


def note_density_similarity(x, y):
    nd_sim = []
    for i in range(len(x)):
        x_nd = len(x[i])
        y_nd = len(y[i])
        nd_sim_tmp = 1 - abs(x_nd - y_nd) / 48
        nd_sim.append(nd_sim_tmp)
    return nd_sim


def chroma_similarity(x, y):
    """
    paper: https://arxiv.org/abs/2105.04090
    """
    chroma_sim = []
    for i in range(len(x)):
        x_r0 = np.zeros(12)
        x_r1 = np.zeros(12)
        y_r0 = np.zeros(12)
        y_r1 = np.zeros(12)
        for bar, position, track, pitch, duration in x[i]:
            if position < 24:
                x_r0[pitch % 12] = 1
            else:
                x_r1[pitch % 12] = 1
        for bar, position, track, pitch, duration in y[i]:
            if position < 24:
                y_r0[pitch % 12] = 1
            else:
                y_r1[pitch % 12] = 1
        cs = 0
        if np.linalg.norm(x_r0) != 0 and np.linalg.norm(y_r0) != 0:
            cs += np.dot(x_r0, y_r0) / (np.linalg.norm(x_r0) * np.linalg.norm(y_r0))
        if np.linalg.norm(x_r1) != 0 and np.linalg.norm(y_r1) != 0:
            cs += np.dot(x_r1, y_r1) / (np.linalg.norm(x_r1) * np.linalg.norm(y_r1))
        chroma_sim.append(cs / 2)
    return chroma_sim


def grooving_similarity(x, y):
    """
    paper: https://arxiv.org/abs/2105.04090
    """
    grooving_sim = []
    for i in range(len(x)):
        x_r = np.zeros(48)
        y_r = np.zeros(48)
        for bar, position, track, pitch, duration in x[i]:
            x_r[position] += 1
        for bar, position, track, pitch, duration in y[i]:
            y_r[position] += 1
        if np.linalg.norm(x_r) == 0 or np.linalg.norm(y_r) == 0:
            grooving_sim.append(0)
            continue
        cos_sim = np.dot(x_r, y_r) / (np.linalg.norm(x_r) * np.linalg.norm(y_r))
        grooving_sim.append(cos_sim)
    return grooving_sim


def _entropy(prob):
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.nansum(prob * np.log2(prob))


def pitch_class_entropy(x):
    pce = []
    for i in range(len(x)):
        counter = np.zeros(12)
        for bar, position, track, pitch, duration in x[i]:
            counter[pitch % 12] += 1
        if np.sum(counter) == 0:
            continue
        counter = counter / np.sum(counter)
        pce.append(_entropy(counter))
    return pce


if __name__ == "__main__":
    x = [[[1, 6, 0, 60, 24], [1, 6, 24, 62, 48]], [[1, 8, 0, 65, 48]]]
    y = [
        [[1, 6, 0, 60, 24], [1, 6, 24, 62, 48], [1, 8, 0, 65, 48]],
        [[1, 8, 48, 67, 48]],
    ]
    print(f"note_f1: {note_f1(x, y)}")
    print(f"pianoroll_f1: {pianoroll_f1(x, y)}")
    print(f"pitch_range_similarity: {pitch_range_similarity(x, y)}")
    print(f"note_density_similarity: {note_density_similarity(x, y)}")
    print(f"chroma_similarity: {chroma_similarity(x, y)}")
    print(f"grooving_similarity: {grooving_similarity(x, y)}")
    print(f"pitch_class_entropy(x): {pitch_class_entropy(x)}")
    print(f"pitch_class_entropy(y): {pitch_class_entropy(y)}")
