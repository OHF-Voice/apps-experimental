ARG BUILD_FROM
FROM ${BUILD_FROM}
ARG BUILD_ARCH

# Set shell
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install
WORKDIR /usr/src
ENV PIP_BREAK_SYSTEM_PACKAGES=1

COPY requirements.txt ./

RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        netcat-traditional \
        python3 \
        python3-pip \
        python3-dev \
        build-essential \
        cmake \
        ninja-build \
        libgomp1 \
    && pip3 install --no-cache-dir -U \
        setuptools \
        wheel \
    # Compile llama-cpp-python from source to maximize CPU optimizations.
    && CMAKE_ARGS='-DCMAKE_BUILD_TYPE=Release' \
       pip3 install --no-cache-dir --no-binary llama-cpp-python llama-cpp-python \
    && pip3 install --no-cache-dir -r ./requirements.txt \
    && rm -rf /var/lib/apt/lists/*

COPY src/templates/*.html ./templates/
COPY src/intent-tools.yaml ./intent-tools.yaml
COPY src/benchmark.yaml ./benchmark.yaml
COPY src/*.py ./

WORKDIR /
COPY rootfs /

HEALTHCHECK --start-period=10m \
    CMD echo '{ "type": "describe" }' \
        | nc -w 1 localhost 10500 \
        | grep -iq "intent-gemma4" \
        || exit 1
