#!/bin/bash

set -e

WORKING_DIR="$(mktemp -d)"
TRIMMOMATIC="java -jar /usr/local/Trimmomatic-0.39/trimmomatic-0.39.jar"

R1_ADAPTER="CTGTCTCTTATACACATCTCCGAGCCCACGAGAC"
R2_ADAPTER="CTGTCTCTTATACACATCTGACGCTGCCGACGA"
FW_PRIMER="GAAATACCTATTGCCTACGGC"
RV_PRIMER="GACGTTCCGGACTACGGTTCC"
VECTER_SEQ="CACCACCATCACCATCACTAGTACCCGTACGACGTTCCGGACTACGGTTCC"
VECTOR_STOP_CODON_DOWNSTREAM="TAGTACCCGTACGACGTTCCGGACTACGGTTCC"

function print_usage_and_exit() {
	msg=$1

	echo >&2 "Usage: construct_VHHs.sh <R1.fastq.gz file> <R2.fastq.gz file> <Library name>"
	if [ ${#msg} -ge 1 ]; then
		echo >&2 "Error: $msg"
	fi
	exit 1
}

# Check args
[ $# -ne 3 ] && print_usage_and_exit

R1_FILE="$(realpath $1)"
R2_FILE="$(realpath $2)"
LIBRARY_NAME="$3"
[ ! -e "$R1_FILE" ] && print_usage_and_exit "R1 File does not exists"
[ ! -e "$R2_FILE" ] && print_usage_and_exit "R2 File does not exists"
[ -z "$LIBRARY_NAME" ] && print_usage_and_exit "Library name required"

cd $WORKING_DIR

# Adapter trimming
cutadapt --quiet --match-read-wildcards -O 1 -e 0.2 -a $R1_ADAPTER $R1_FILE -o R1_cutadapt.fastq
cutadapt --quiet --match-read-wildcards -O 1 -e 0.2 -a $R2_ADAPTER $R2_FILE -o R2_cutadapt.fastq

# Quality trimming
$TRIMMOMATIC PE -phred33 -trimlog log.txt R1_cutadapt.fastq R2_cutadapt.fastq paired_trimmomatic_output_1.fastq unpaired_trimmomatic_output_1.fastq paired_trimmomatic_output_2.fastq unpaired_trimmomatic_output_2.fastq LEADING:0 TRAILING:0 SLIDINGWINDOW:20:20 MINLEN:50

# Merge
fastq-join -p 8 -m 6 paired_trimmomatic_output_1.fastq paired_trimmomatic_output_2.fastq -o merged.%.fastq &> /dev/null

# Pickup VHH Sequence
# Extract sequences that have a perfect match to both the forward primer sequence at the 5’ end and
# the reverse primer sequence at the 3’ end of the amplicon sequence. These sequences are amplified
# specifically by PCR, including the VHH region, during sample preparation and only the amplified
# region is extracted.
seqkit grep -sp $FW_PRIMER merged.join.fastq > out_rm_fw.fastq
seqkit grep -sp $RV_PRIMER out_rm_fw.fastq > out_rm_fw_rv.fastq
seqkit grep -sp $VECTER_SEQ out_rm_fw_rv.fastq > out_vector.fastq
cutadapt --quiet -e 0 -g $FW_PRIMER out_vector.fastq > out_vector_rm5amp.fastq

# Convert fastq to fasta
seqkit fq2fa out_vector_rm5amp.fastq > out_vector_rm5amp.fasta

# Add two bases of “AT” to the 5’ end of the amplicon sequence.
seqkit mutate -i "0:AT$FW_PRIMER" --quiet out_vector_rm5amp.fasta > out_vector_rs5amp.fasta

# Trim off the sequence downstream of the stop codon included in the 3’ end vector sequence.
cutadapt --quiet -e 0 -a $VECTOR_STOP_CODON_DOWNSTREAM out_vector_rs5amp.fasta > out_vector_rm3flank.fasta

# Convert the nucleotide sequence to an amino acid sequence.
transeq -sequence out_vector_rm3flank.fasta -outseq translated.fasta
seqkit seq -w 0 translated.fasta > translated_complete.fasta

# Remove all nucleotide bases after the stop codon.
# (In transeq, stop codons are converted to an asterisk symbol “*”)
gawk '{ sub("*.*$",""); print $0 > "translated_trimtillstop.fasta";}' translated_complete.fasta

# Extract sequences that contain a His-tag sequence (6 residues of H) at the C-terminus.
seqkit grep -w 0 -sp HHHHHH translated_trimtillstop.fasta > translated_trimtillstop_His.fasta
gawk '{ sub("HHHHHH.*$",""); print $0 > "translated_trimtillstop_His_cut.fasta";}' translated_trimtillstop_His.fasta
seqkit mutate -i -1:HHHHHH --quiet translated_trimtillstop_His_cut.fasta > translated_trimtillstop_His6.fasta

# Remove all sequences from the FASTA file that contain ambiguous amino acid “X”.
seqkit grep -v -sp X translated_trimtillstop_His6.fasta > translated_trimtillstop_His_clean.fasta

# Limit the FASTA file to sequences that are 100 or more amino acids in length.
seqkit seq -w 0 -m 100 translated_trimtillstop_His_clean.fasta > translated_trimtillstop_His_complete.fasta

# Append the library name to the FASTA file.
gawk '{ if(NR%2==1){ sub("$",";sample='$LIBRARY_NAME'")}; print $0;}' translated_trimtillstop_His_complete.fasta

rm -rf $WORKING_DIR
