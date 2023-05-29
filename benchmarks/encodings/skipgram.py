# This is implemented based on PIPR's results.
# Ref: https://github.com/muhaochen/seq_ppi/tree/master/embeddings
import os
import argparse
import numpy as np
import pandas as pd

MAX_LENGTH_AB = 180
MAX_LENGTH_AG = 220
N_ACIDS = 20
acids_dict = {
    "A": [
        -0.17691335,
        -0.19057421,
        0.045527875,
        -0.175985,
        1.1090639,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    "C": [
        -0.31572455,
        0.38517416,
        0.17325026,
        0.3164464,
        1.1512344,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ],
    "D": [
        0.00600859,
        -0.1902303,
        -0.049640052,
        0.15067418,
        1.0812483,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ],
    "E": [
        -0.06940994,
        -0.34011552,
        -0.17767446,
        0.251,
        1.0661993,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ],
    "F": [
        0.2315121,
        -0.01626652,
        0.25592703,
        0.2703909,
        1.0793934,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    "G": [
        -0.07281224,
        0.01804472,
        0.22983849,
        -0.045492448,
        1.1139168,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    "H": [
        0.019046513,
        -0.023256639,
        -0.06749539,
        0.16737276,
        1.0796973,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
    ],
    "I": [
        0.15077977,
        -0.1881559,
        0.33855876,
        0.39121667,
        1.0793937,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    "K": [
        0.22048187,
        -0.34703028,
        0.20346786,
        0.65077996,
        1.0620389,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
    ],
    "L": [
        0.0075188675,
        -0.17002057,
        0.08902198,
        0.066686414,
        1.0804346,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    "M": [
        0.06302169,
        -0.10206237,
        0.18976009,
        0.115588315,
        1.0927621,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    "N": [
        0.41597384,
        -0.22671205,
        0.31179032,
        0.45883527,
        1.0529875,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
    ],
    "P": [
        0.017954966,
        -0.09864355,
        0.028460773,
        -0.12924117,
        1.0974121,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    "Q": [
        0.25189143,
        -0.40238172,
        -0.046555642,
        0.22140719,
        1.0362468,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
    ],
    "R": [
        -0.15621762,
        -0.19172126,
        -0.209409,
        0.026799612,
        1.0879921,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
    ],
    "S": [
        0.17177454,
        -0.16769698,
        0.27776834,
        0.10357749,
        1.0800852,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    "T": [
        0.054446213,
        -0.16771607,
        0.22424258,
        -0.01337227,
        1.0967118,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    "V": [
        -0.09511698,
        -0.11654304,
        0.1440215,
        -0.0022315443,
        1.1064949,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    "W": [
        0.25281385,
        0.12420933,
        0.0132171605,
        0.09199735,
        1.0842415,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
    ],
    "Y": [
        0.27962074,
        -0.051454283,
        0.114876375,
        0.3550331,
        1.0615551,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
}


def fill_zeros(raw_data, max_length):
    filled_data = []
    for seq in raw_data:
        if len(seq) < max_length:
            seq = list(seq) + ["0"] * (max_length - len(seq))
        else:
            raise ValueError("Sequence length is longer than max_length.")
        filled_data.append(seq)
    return np.array(filled_data)


def acids_to_vecs(data, acids_dict):
    vecs = np.zeros((data.shape[0], data.shape[1], 12), dtype=np.float16)
    for i, seq in enumerate(data):
        for j, acid in enumerate(seq):
            vecs[i][j] = acids_dict[acid]
    return vecs.transpose((0, 2, 1))


def main(args):
    all_data = pd.read_csv(args.data_path)
    acids_dict["0"] = [0] * 12

    all_ab = acids_to_vecs(
        fill_zeros(all_data["VHH_sequence"].values, MAX_LENGTH_AB), acids_dict
    ).astype(np.float16)
    all_ag = acids_to_vecs(
        fill_zeros(all_data["Ag_sequence"].values, MAX_LENGTH_AG), acids_dict
    ).astype(np.float16)

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
        "--data-path",
        type=str,
        required=True,
        help="Path of the target CSV file to be preprocessed",
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
