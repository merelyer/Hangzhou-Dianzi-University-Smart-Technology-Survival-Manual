from pathlib import Path
import csv
import time

import numpy as np
ROOT = Path(r"C:\Users\lottttt\Documents\Codex\2026-06-02\files-mentioned-by-the-user-2023")
REPO = ROOT / "work" / "RanPAC-main"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def onehot(labels, num_classes):
    y = np.zeros((len(labels), num_classes), dtype=np.float32)
    y[np.arange(len(labels)), labels] = 1.0
    return y


def select_per_class(features, labels, classes, samples_per_class):
    xs, ys = [], []
    for cls in classes:
        idx = np.where(labels == cls)[0][:samples_per_class]
        xs.append(features[idx])
        ys.append(labels[idx])
    return np.concatenate(xs), np.concatenate(ys)


def make_synthetic_incremental_data(rng, num_classes, input_dim, train_per_class, test_per_class):
    prototypes = rng.standard_normal((num_classes, input_dim)).astype(np.float32)
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
    train_x, train_y, test_x, test_y = [], [], [], []
    for cls in range(num_classes):
        train_x.append(prototypes[cls] + 0.25 * rng.standard_normal((train_per_class, input_dim)).astype(np.float32))
        test_x.append(prototypes[cls] + 0.25 * rng.standard_normal((test_per_class, input_dim)).astype(np.float32))
        train_y.extend([cls] * train_per_class)
        test_y.extend([cls] * test_per_class)
    return (
        np.concatenate(train_x).astype(np.float32),
        np.array(train_y),
        np.concatenate(test_x).astype(np.float32),
        np.array(test_y),
    )


def main():
    started = time.time()
    seed = 1993
    rng = np.random.default_rng(seed)
    class_order = rng.permutation(100)

    tasks = 2
    classes_per_task = 10
    samples_per_class_train = 80
    samples_per_class_test = 40
    rp_dim = 512
    ridge = 1.0
    input_dim = 768
    total_classes = tasks * classes_per_task

    train_x, mapped_train_y, test_x, mapped_test_y = make_synthetic_incremental_data(
        rng, total_classes, input_dim, samples_per_class_train, samples_per_class_test
    )

    # Random projection matrix, fixed across tasks.
    w_rand = rng.standard_normal((input_dim, rp_dim)).astype(np.float32) / np.sqrt(input_dim)
    q = np.zeros((rp_dim, tasks * classes_per_task), dtype=np.float32)
    g = np.zeros((rp_dim, rp_dim), dtype=np.float32)

    rows = []
    for task in range(tasks):
        seen_classes = np.arange((task + 1) * classes_per_task)
        new_classes = np.arange(task * classes_per_task, (task + 1) * classes_per_task)

        x_new, y_new = select_per_class(
            train_x, mapped_train_y, new_classes, samples_per_class_train
        )
        h_new = np.maximum(x_new @ w_rand, 0.0)
        y_oh = onehot(y_new, tasks * classes_per_task)
        q += h_new.T @ y_oh
        g += h_new.T @ h_new

        weights = np.linalg.solve(g + ridge * np.eye(rp_dim, dtype=np.float32), q).T

        x_eval, y_eval = select_per_class(
            test_x, mapped_test_y, seen_classes, samples_per_class_test
        )
        h_eval = np.maximum(x_eval @ w_rand, 0.0)
        logits = h_eval @ weights[: len(seen_classes)].T
        pred = logits.argmax(axis=1)
        acc = (pred == y_eval).mean() * 100.0
        rows.append(
            {
                "task": task + 1,
                "seen_classes": len(seen_classes),
                "train_samples_this_task": len(y_new),
                "test_samples_seen_classes": len(y_eval),
                "top1_accuracy": round(float(acc), 2),
            }
        )

    out_csv = OUT / "ranpac_core_quick_results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    out_txt = OUT / "ranpac_core_quick_results_summary.txt"
    with out_txt.open("w", encoding="utf-8") as f:
        f.write("RanPAC core quick local experiment on synthetic class-incremental data\n")
        f.write(f"seed: {seed}\n")
        f.write(f"tasks: {tasks}, classes_per_task: {classes_per_task}\n")
        f.write(f"train samples/class: {samples_per_class_train}\n")
        f.write(f"test samples/class: {samples_per_class_test}\n")
        f.write(f"feature dimension: {input_dim}\n")
        f.write(f"random projection dimension M: {rp_dim}\n")
        f.write(f"ridge lambda: {ridge}\n")
        f.write(f"runtime_seconds: {time.time() - started:.2f}\n\n")
        for row in rows:
            f.write(
                f"Task {row['task']}: seen {row['seen_classes']} classes, "
                f"Top-1 accuracy {row['top1_accuracy']}%\n"
            )

    print(out_csv)
    for row in rows:
        print(row)
    print(f"runtime_seconds={time.time() - started:.2f}")


if __name__ == "__main__":
    main()
