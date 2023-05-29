#!/bin/bash

set -eux

REPO_ROOT="$(git rev-parse --show-toplevel)"
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
DATA_DIR="$REPO_ROOT/data"
OUT_DIR="$REPO_ROOT/out"
mkdir -p $OUT_DIR

IMAGE="avida-hil6:latest"
VHH_CONSTRUCTOR="docker run --rm -v $DATA_DIR:/work/data -v $PWD/out:/work/out $IMAGE"
LIBRARY_LIST="$SCRIPT_DIR/library_list.tsv"

# Construct VHH Sequences
tail -n +2 "$LIBRARY_LIST" | while IFS=$'\t' read -r library_name library_group fastq_r1 fastq_r2; do
	# in container
	$VHH_CONSTRUCTOR bash -c "construct_VHHs.sh /work/data/$fastq_r1 /work/data/$fastq_r2 $library_name > /work/out/$library_name.fasta"
done

# Create library table(cross tabulation)
# - Wild types
$SCRIPT_DIR/create_unique_libtable.py $OUT_DIR/{Mother,Serum,IL-6_WT}-*.fasta > $OUT_DIR/libtable_IL-6_WTs.tsv
# - Mutants
mutant_groups=$(tail -n +2 "$LIBRARY_LIST" | grep -vE '^(Mother|Serum|IL-6_WT)' | awk '{print $2}' | uniq)
for group in $mutant_groups; do
	$SCRIPT_DIR/create_unique_libtable.py $OUT_DIR/{Mother,Serum,$group}-*.fasta > $OUT_DIR/libtable_$group.tsv
done

# Annotation
echo "VHH_sequence,label,Ag_label,Ag_sequence" >> $OUT_DIR/il6_aai_dataset.csv
# - Wild Types
$SCRIPT_DIR/annotate.py $OUT_DIR/libtable_IL-6_WTs.tsv IL-6_WTs >> $OUT_DIR/il6_aai_dataset.csv
# - Mutants
for group in $mutant_groups; do
	$SCRIPT_DIR/annotate.py $OUT_DIR/libtable_$group.tsv $group >> $OUT_DIR/il6_aai_dataset.csv
done
