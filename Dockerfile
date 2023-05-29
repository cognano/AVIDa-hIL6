FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# fastq-join(ea-utils), transeq(emboss)
RUN apt-get update \
    && apt-get install -y curl unzip cutadapt emboss ea-utils openjdk-17-jdk default-jre gawk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Trimmomatic
RUN curl -sL https://github.com/usadellab/Trimmomatic/files/5854859/Trimmomatic-0.39.zip -o /tmp/Trimmomatic.zip \
    && unzip /tmp/Trimmomatic.zip -d /usr/local/ \
    && rm -f /tmp/Trimmomatic.zip

# seqkit
RUN cd /tmp \
    && curl -sLO https://github.com/shenwei356/seqkit/releases/download/v2.2.0/seqkit_linux_amd64.tar.gz \
    && tar -zxvf /tmp/seqkit_linux_amd64.tar.gz \
    && rm -f /tmp/seqkit_linux_amd64.tar.gz \
    && chmod +x /tmp/seqkit \
    && mv /tmp/seqkit /usr/local/bin

# fastqc
RUN curl -sL http://www.bioinformatics.babraham.ac.uk/projects/fastqc/fastqc_v0.11.9.zip -o /tmp/fastqc.zip \
    && unzip /tmp/fastqc.zip -d /usr/local/ \
    && rm -f /tmp/fastqc.zip \
    && chmod +x /usr/local/FastQC/fastqc \
    && ln -s /usr/local/FastQC/fastqc /usr/local/bin/

COPY ./dataset/construct_VHHs.sh /usr/local/bin
