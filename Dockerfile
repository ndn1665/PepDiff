# CUDA가 사전 설치된 NVIDIA 공식 이미지를 베이스로 사용
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# 시스템 패키지 설치 및 타임존 설정
ENV TZ=Asia/Seoul
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    vim \
    nano \
    build-essential \
    python3-dev \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Miniconda 설치
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda init

# Conda PATH 설정
ENV PATH /opt/conda/bin:$PATH

# 작업 디렉토리 설정
WORKDIR /app

# environment.yml 파일 복사
COPY codes/environment.yml .

# Conda Terms of Service 동의
RUN conda config --set channel_priority flexible
RUN conda config --add channels conda-forge
RUN conda config --add channels pytorch
RUN conda config --add channels nvidia

# Anaconda Terms of Service 동의
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Conda 환경 생성
RUN conda env create -f environment.yml

# 기본 쉘을 Conda 환경이 활성화된 bash로 설정
SHELL ["conda", "run", "-n", "pepchain", "/bin/bash", "-c"]

# 기본 Conda 환경 활성화
RUN echo "conda activate pepchain" >> ~/.bashrc

# 컨테이너 실행 시 bash 쉘 시작
CMD ["/bin/bash"]
