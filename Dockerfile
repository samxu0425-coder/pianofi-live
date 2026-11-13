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
# TensorRT 10.4 libraries for ONNX Runtime's TensorRT provider (the engine's --trt): the same
# version as on Windows, NVIDIA's own package, libraries only.
ARG TRT_VERSION=10.4.0
RUN apt-get update \
 && apt-get install -y --no-install-recommends unzip \
 && curl -fsSL -o /tmp/trt.whl https://pypi.nvidia.com/tensorrt-cu12-libs/tensorrt_cu12_libs-${TRT_VERSION}-py2.py3-none-manylinux_2_17_x86_64.whl \
 && unzip -q /tmp/trt.whl 'tensorrt_libs/*' -d /opt/tensorrt \
 && rm /tmp/trt.whl \
 && apt-get purge -y unzip \
 && rm -rf /var/lib/apt/lists/*
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib:/opt/tensorrt/tensorrt_libs

# Build stage: compiler and CMake stay out of the final image.
FROM base AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ make cmake \
 && rm -rf /var/lib/apt/lists/*
COPY cpp /src/cpp
RUN cmake -S /src/cpp -B /build -DONNXRUNTIME_DIR=/opt/onnxruntime \
 && cmake --build /build -j \
 && /build/pf_tests

# Service image (docker build --target service): the engine plus the web server that
# runs one engine per connected page. See RUN.md step 11. Mount /work/models and
# /work/songs; it listens on 8000 (page) and 8001 (events).
FROM base AS service
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-pip \
 && rm -rf /var/lib/apt/lists/* \
 && pip3 install --no-cache-dir websockets==14.1 soundfile==0.12.1
COPY --from=build /build/pianofi_live /app/
COPY ui /app/ui
ENV PIANOFI_ENGINE=/app/pianofi_live
WORKDIR /work
EXPOSE 8000 8001
CMD ["python3", "/app/ui/server.py", "--host", "0.0.0.0", "--songs", "/work/songs"]

# Engine only (the default target).
FROM base
COPY --from=build /build/pianofi_live /build/pf_tests /build/bench_ring /app/
WORKDIR /work
ENTRYPOINT ["/app/pianofi_live"]
