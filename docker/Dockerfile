#FROM nvidia/cuda:12.0.1-devel-ubuntu22.04
# FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y libtbb-dev libboost-all-dev libfmt-dev libtbb-dev libopenblas-dev libblas-dev\
    build-essential \
    ninja-build \
    git wget 

RUN wget https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-Linux-x86_64.sh \
    -q -O /tmp/cmake-install.sh \
    && chmod u+x /tmp/cmake-install.sh \
    && mkdir /opt/cmake-3.26.4 \
    && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.26.4 \
    && rm /tmp/cmake-install.sh \
    && ln -s /opt/cmake-3.26.4/bin/* /usr/local/bin

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update
RUN apt-get install -y vim
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*
# RUN apt-get install -y openssh-server

ENV PATH="/root/miniconda3/bin:${PATH}"
RUN wget -q\
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version
RUN conda install pip
RUN conda init bash
RUN /bin/bash -c "source /root/.bashrc"
RUN conda install -y numpy scipy pandas seaborn -c conda-forge

# RUN git clone https://github.com/llvm/llvm-project.git \
#     && cd llvm-project \
#     && git checkout release/17.x 

# RUN mkdir llvm-project/build \
#     && cd llvm-project/build \
#     && cmake -G Ninja -DCMAKE_BUILD_TYPE="Release" -DLLVM_ENABLE_PROJECTS="clang" -DLLVM_TARGETS_TO_BUILD="X86" \
#         -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi" -DLLVM_ENABLE_RTTI=ON ../llvm \
#     && ninja \
#     && ninja install -j4

# RUN llvm-as --version \
#     && clang --version

# RUN apt-get update -y
ENV PATH="/X-GMiner:${PATH}"
WORKDIR /home/wuzhenlin/workspace/2-graphmining/

