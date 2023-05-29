# CKSAAP is implemented based on AbAgIntPre's implementation.
# Ref: https://github.com/emersON106/AbAgIntPre/blob/main/model/sequence%20encoding.py
import os
import argparse
import numpy as np
import pandas as pd
from itertools import product

AA = [
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
DP = list(product(AA, AA))
DP_list = []
for i in DP:
    DP_list.append(str(i[0]) + str(i[1]))


def returnCKSAAPcode(query_seq, k=3):
    code_final = []
    for turns in range(k + 1):
        DP_dic = {}
        code = []
        for i in DP_list:
            DP_dic[i] = 0
        for i in range(len(query_seq) - turns - 1):
            tmp_dp_1 = query_seq[i]
            tmp_dp_2 = query_seq[i + turns + 1]
            tmp_dp = tmp_dp_1 + tmp_dp_2
            if tmp_dp in DP_dic.keys():
                DP_dic[tmp_dp] += 1
            else:
                DP_dic[tmp_dp] = 1
        for i, j in DP_dic.items():
            code.append(j / (len(query_seq) - turns - 1))
        code_final += code
    return code_final


def main(args):
    all_data = pd.read_csv(args.data_path)
    all_ab = []
    for vhh_seq in all_data["VHH_sequence"].values:
        all_ab.append(returnCKSAAPcode(vhh_seq))
    all_ab = np.array(all_ab).astype(np.float16).reshape([-1, 4, 20, 20])
    all_ag = []
    for ag_seq in all_data["Ag_sequence"].values:
        all_ag.append(returnCKSAAPcode(ag_seq))
    all_ag = np.array(all_ag).astype(np.float16).reshape([-1, 4, 20, 20])

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
