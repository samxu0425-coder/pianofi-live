# PianoFi Live engine on Linux with an NVIDIA GPU. See RUN.md step 8.
#
#   docker build -t pianofi-live .
#
# The image holds only the engine. The model files, the songs and the output
# folder are mounted when it runs, so the image has no weights in it.

# CUDA 12 + cuDNN 9: what ONNX Runtime 1.20 needs for the GPU.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base
ARG ORT_VERSION=1.20.1
# Same ONNX Runtime version as the Windows build, Linux GPU release.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && curl -fsSL https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz \
    | tar -xz -C /opt \
 && mv /opt/onnxruntime-linux-x64-gpu-${ORT_VERSION} /opt/onnxruntime \
 && rm -rf /var/lib/apt/lists/*
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib

# Build stage: compiler and CMake stay out of the final image.
FROM base AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ make cmake \
 && rm -rf /var/lib/apt/lists/*
COPY cpp /src/cpp
RUN cmake -S /src/cpp -B /build -DONNXRUNTIME_DIR=/opt/onnxruntime \
 && cmake --build /build -j \
 && /build/pf_tests

FROM base
COPY --from=build /build/pianofi_live /build/pf_tests /build/bench_ring /app/
WORKDIR /work
ENTRYPOINT ["/app/pianofi_live"]
