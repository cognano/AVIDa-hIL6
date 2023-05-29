import os
import argparse
import numpy as np
import pandas as pd

MAX_LENGTH_AB = 180
MAX_LENGTH_AG = 220
N_ACIDS = 20
acids = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]


def fill_zeros(raw_data, max_length):
    filled_data = []
    for seq in raw_data:
        if len(seq) < max_length:
            seq = list(seq) + ["0"] * (max_length - len(seq))
        else:
            raise ValueError("Sequence length is longer than max_length.")
        filled_data.append(seq)
    return np.array(filled_data)


def acids_to_one_hot(data, acids_dict, n_acids):
    one_hot_vec = np.zeros((data.shape[0], data.shape[1], n_acids), dtype=np.int8)
    for i, seq in enumerate(data):
        for j, acid in enumerate(seq):
            one_hot_vec[i][j] = acids_dict[acid]
    return one_hot_vec.transpose((0, 2, 1))


def main(args):
    all_data = pd.read_csv(args.data_path)
    acids_dict = {}
    for i, acid in enumerate(acids):
        vec = np.zeros(N_ACIDS, dtype=np.int8)
        vec[i] = 1
        acids_dict[acid] = vec
    acids_dict["0"] = np.zeros(N_ACIDS, dtype=np.int8)

    # Covert amino acid sequences into one-hot vectors
    all_ab = acids_to_one_hot(
        fill_zeros(all_data["VHH_sequence"].values, MAX_LENGTH_AB), acids_dict, N_ACIDS
    ).astype(np.int8)
    all_ag = acids_to_one_hot(
        fill_zeros(all_data["Ag_sequence"].values, MAX_LENGTH_AG), acids_dict, N_ACIDS
    ).astype(np.int8)

    all_label = all_data["label"].values.astype("int8")
    np.savez_compressed(
        os.path.join(args.save_dir, args.file_name + ".npz"),
        ab=all_ab,
        ag=all_ag,
        label=all_label,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path of the target CSV file to be preprocessed"
    )
    parser.add_argument(
        "--save-dir",
        default=".",
        type=str,
        help="Directory to save the preprocessed file (default: .)",
    )
    parser.add_argument(
        "--file-name",
        default="il6_aai_dataset",
        type=str,
        help="Name of output npz file (default: il6_aai_dataset)",
    )
    args = parser.parse_args()
    main(args)
