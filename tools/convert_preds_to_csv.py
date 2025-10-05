import os
import os.path as osp
import csv
import numpy as np


def convert_dir(pred_dir: str) -> None:
    if not osp.isdir(pred_dir):
        print(f"No directory: {pred_dir}")
        return
    for name in sorted(os.listdir(pred_dir)):
        if not name.lower().endswith('.npy'):
            continue
        npy_path = osp.join(pred_dir, name)
        csv_path = osp.splitext(npy_path)[0] + '.csv'
        arr = np.load(npy_path)
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t', 'value'])
            for i, v in enumerate(arr.tolist()):
                w.writerow([i, float(v)])
        print(f"Wrote {csv_path}")


if __name__ == '__main__':
    root = osp.join(osp.dirname(osp.dirname(__file__)), 'outputs', 'mdar', 'predictions')
    convert_dir(root)


