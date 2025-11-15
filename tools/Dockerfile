FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime AS base

LABEL maintainer="jgjxzqw@gmail.com"

RUN apt-get update && \
    apt-get install -y --no-install-recommends tzdata locales && \
    ln -snf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    locale-gen en_US.UTF-8 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y \
    wget \
    unzip \
    curl \
    cmake \
    jq && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install pandas numpy xgboost scikit-learn seaborn

WORKDIR /workspace
