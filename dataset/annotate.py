#!/usr/bin/env python3

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
import scipy.stats as st
from constants import AG_SEQUENCES, INDEX_COLUMN_NAME, INTERACTION


def load_library_table(filepath):
    df = pd.read_table(filepath, sep="\t", index_col=INDEX_COLUMN_NAME)
    return df


def load_sequence(filepath):
    df_seq = pd.read_table(filepath, sep="\t", header=None, usecols=[0, 1])

    df_seq_f = df_seq[0].str.split(";", expand=True)
    df_seq_f = df_seq_f.drop(df_seq_f.columns[2], axis="columns")
    df_seq_f.columns = ["#ONU ID", "size"]

    df_seq_f = df_seq_f.set_index("#ONU ID")
    df_seq_f["size"] = df_seq_f["size"].str.replace("size=", "")
    df_seq_f["sequence"] = df_seq[1].to_numpy()

    return df_seq_f


def list_library_groups(libraries, prefix):
    target_libs = [lib for lib in libraries if lib.startswith(prefix)]
    stripped_suffix = ["-".join(lib.split("-")[:-1]) for lib in target_libs]
    target_lib_groups = list(set(stripped_suffix))

    return target_lib_groups


def create_p_val_df(df):
    total_reads_by_lib = df.sum(numeric_only=True).to_dict()
    df_read_ratio = df / df.sum()

    idx = df.index.to_list()
    target_libs = df.filter(regex="^(?!Mother-).*$").columns.to_list()

    increase = {}
    p_value = {}
    for lib in target_libs:
        suffix = lib.split("-")[-1]
        mother_total = total_reads_by_lib[f"Mother-{suffix}"]
        target_total = total_reads_by_lib[lib]

        inc = np.where(df_read_ratio[lib] > df_read_ratio[f"Mother-{suffix}"], True, False)
        p = np.vectorize(calc_p_val)(df[f"Mother-{suffix}"], df[lib], mother_total, target_total)

        increase[lib] = dict(zip(idx, inc))
        p_value[lib] = dict(zip(idx, p))

    p_val_revised = {}
    for lib, p_dict in p_value.items():
        p_val_revised[lib] = {}
        for k, v in p_dict.items():
            p_val_revised[lib][k] = v if increase[lib][k] else -v

    df_p_val = pd.DataFrame(data=p_val_revised)
    return df_p_val


def calc_p_val(mother_onu, target_onu, mother_total, target_total):
    if mother_onu == 0 and target_onu == 0:
        return np.nan
    p_1 = mother_onu / mother_total
    p_2 = target_onu / target_total
    pooled_p = (mother_onu + target_onu) / (mother_total + target_total)
    z_score = (p_1 - p_2) / math.sqrt(
        pooled_p * (1 - pooled_p) * (1 / mother_total + 1 / target_total)
    )
    z_score = abs(z_score)
    return st.norm.cdf(-z_score) + (1 - st.norm.cdf(z_score))


def annotate(df_p_val, background_group, target_groups, sig_level):
    df_background = df_p_val.filter(like=background_group, axis="columns").copy()
    df_background = select_sample(df_background)

    if len(target_groups) > 1:
        target_dfs = []
        for grp in target_groups:
            df_target = df_p_val.filter(like=grp, axis="columns").copy()
            target_dfs.append(select_sample(df_target))
        merged_df = pd.concat(target_dfs, axis=1)
        samples = merged_df["sample"].apply(avg_p_val, axis=1)
    else:
        df_target = df_p_val.filter(like=target_groups[0], axis="columns").copy()
        samples = select_sample(df_target)["sample"]

    labeled = np.vectorize(identify_interactions)(
        df_background["sample"].fillna(-1),
        samples.fillna(-1),
        sig_level,
    )

    labeled_dict = dict(zip(df_p_val.index.to_list(), labeled))
    df_labeled = pd.DataFrame(data=labeled_dict, index=["label"]).T
    return df_labeled


def avg_p_val(row):
    score_list = [calc_score(val) for val in row]
    avg = sum(score_list) / len(score_list)
    return 10 ** avg if avg <= 0 else (-1) * 10 ** (-1 * avg)


def calc_score(val):
    inc = False if val <= 0 else True
    if math.isnan(val):
        return 0
    elif val == 0.000000e00:
        return -10 if inc else 10
    return math.log10(val) if inc else -math.log10(-val)


def select_sample(df):
    df["sample"] = df.apply(select_p_val, axis=1)
    return df


def select_p_val(row):
    if row.max() > 0:
        return row[row > 0].min()
    else:
        return row[row <= 0].max()


def identify_interactions(background, target, threshold):
    if abs(target) > threshold:
        return INTERACTION.NON_SIGNIFICANT

    if target <= 0:
        return INTERACTION.NON_BINDER

    if abs(background) <= threshold and background > 0:
        return INTERACTION.NOISE

    if background > 0 and background / target < 10 ** 2.5:
        return INTERACTION.NON_SIGNIFICANT

    return INTERACTION.BINDER


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("libtable_file", type=str)
    parser.add_argument("antigen_label", type=str)
    parser.add_argument("--background-prefix", type=str, default="Serum")
    parser.add_argument("--target-prefix", type=str, default="IL-6")
    args = parser.parse_args()

    ag_seq = AG_SEQUENCES.get(args.antigen_label, "UNKNWON")

    libtable_path = os.path.abspath(args.libtable_file)
    df_tab = load_library_table(libtable_path)
    target_lib_groups = list_library_groups(
        libraries=df_tab.columns.to_list(), prefix=args.target_prefix
    )

    # Labeling
    df_p_val = create_p_val_df(df_tab.drop(columns=["total", "sequence"]))
    df_label = annotate(df_p_val, args.background_prefix, target_lib_groups, 0.05)
    df_result = pd.concat([df_tab, df_label], axis=1)

    # Adjust output
    df_out = df_result.loc[:, ["sequence", "label"]].rename(columns={"sequence": "VHH_sequence"})
    df_out = df_out.assign(Ag_label=args.antigen_label, Ag_sequence=ag_seq)
    replace_dict = {
        INTERACTION.BINDER: "1",
        INTERACTION.NON_BINDER: "0",
    }
    df_out[df_out["label"].isin([INTERACTION.BINDER, INTERACTION.NON_BINDER])].replace(
        replace_dict
    ).to_csv(sys.stdout, index=False, header=False)


if __name__ == "__main__":
    main()
