#!/usr/bin/env python3

import argparse
import logging
import os
import sys

import pandas as pd
from Bio import SeqIO

LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def load_fasta(filepaths, library_filters=None):
    sequences = []

    for fp in filepaths:
        records = SeqIO.parse(fp, "fasta")
        sequences += [[str(rec.seq), desc_to_lib_name(rec.description)] for rec in records]

    df = pd.DataFrame(data=sequences)
    df = df.set_axis(["sequence", "library"], axis="columns", copy=False)

    if library_filters is not None:
        conditions = []
        for filter in library_filters:
            conditions.append(df["library"].str.contains(filter, case=False, na=False))

        combined_condition = pd.concat(conditions).groupby(level=0).any()
        df = df[combined_condition]

    return df


def desc_to_lib_name(seq_description):
    t = seq_description.split(";")
    t = t[-1].split("=")
    return t[-1]


def filter_by_occurrence(df, min_occurrences):
    seq_counts = df["sequence"].value_counts()
    valid_sequences = seq_counts[seq_counts >= min_occurrences].index
    df_filtered = df[df["sequence"].isin(valid_sequences)]
    return df_filtered


def list_target_lib_groups(df):
    target_lib_groups = list(
        set(["-".join(lib.split("-")[:-1]) for lib in df["library"].unique()])
    )
    target_lib_groups.remove("Mother")
    target_lib_groups.remove("Serum")
    return target_lib_groups


def crosstab(df):
    df_cross = pd.crosstab(df["sequence"], df["library"])
    df_cross["total"] = df_cross.sum(axis="columns")
    df_cross.sort_values("total", ascending=False, inplace=True)

    # set index
    df_cross["#ONU ID"] = [f"ONU{id}" for id in range(1, len(df_cross.index) + 1)]
    df_cross = df_cross.reset_index().set_index("#ONU ID")

    df_result = df_cross.filter(regex="^(?!sequence).*$")
    df_result["sequence"] = df_cross["sequence"]
    return df_result


def main():
    parser = argparse.ArgumentParser(
        description="Cross-tabulate amino acid sequences and libraries."
    )
    parser.add_argument("fasta_files", type=str, nargs="*")
    parser.add_argument(
        "-m",
        "--min-count",
        type=int,
        default=2,
        help="Minimum count of amino acid sequences (default: %(default)s)",
    )
    parser.add_argument(
        "-f",
        "--filter-library",
        action="append",
        type=str,
        help="Specify the name of the library to be filtered. (repeatable)",
    )
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS.keys(),
        default="info",
        help="Set the logging level (default: %(default)s)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=LOG_LEVELS[args.log_level], format="[%(levelname)s] %(message)s")

    fasta_paths = [os.path.abspath(f) for f in args.fasta_files]
    df = load_fasta(fasta_paths)

    libs = df["library"].unique()
    logging.info(f"{len(libs)} libraries loaded: {sorted(libs)}")

    df_cross = crosstab(filter_by_occurrence(df, args.min_count))
    df_cross.to_csv(sys.stdout, sep="\t")


if __name__ == "__main__":
    main()
