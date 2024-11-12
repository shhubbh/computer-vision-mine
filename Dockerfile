# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Update and install required packages
RUN apt-get update && \
    apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    sudo \
    vim \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Mamba
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Set environment variables for Conda and Mamba
ENV PATH /opt/conda/bin:$PATH
RUN conda install -n base -c conda-forge mamba

# Copy all files from the current directory to /workspace in the Docker image
COPY . . 

# Set the working directory
WORKDIR /workspace

# Create a non-root user and switch to that user
RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo
USER docker

# Set the entrypoint
ENTRYPOINT ["/bin/bash"]
