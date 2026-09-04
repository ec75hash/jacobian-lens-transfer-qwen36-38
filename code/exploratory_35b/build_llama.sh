#!/bin/bash
# Build llama.cpp @ pinned commit with the capture_activations example (sm_120). Log: /workspace/logs/build.log
set -euo pipefail
PIN=1772701f9
LLAMA_DIR=/workspace/llama.cpp
export PATH=/usr/local/cuda/bin:$PATH
which cmake >/dev/null 2>&1 || /venv/main/bin/pip install -q cmake ninja
export PATH=/venv/main/bin:$PATH
if [ ! -d "$LLAMA_DIR/.git" ]; then
  git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi
git -C "$LLAMA_DIR" checkout "$PIN"
name=capture_activations
d="$LLAMA_DIR/examples/$name"; mkdir -p "$d"
cp /workspace/src/$name.cpp "$d/$name.cpp"
cat > "$d/CMakeLists.txt" <<EOC
set(TARGET llama-$name)
add_executable(\${TARGET} $name.cpp)
install(TARGETS \${TARGET} RUNTIME)
target_link_libraries(\${TARGET} PRIVATE common llama \${CMAKE_THREAD_LIBS_INIT})
target_compile_features(\${TARGET} PRIVATE cxx_std_17)
EOC
grep -q "add_subdirectory($name)" "$LLAMA_DIR/examples/CMakeLists.txt" || echo "add_subdirectory($name)" >> "$LLAMA_DIR/examples/CMakeLists.txt"
nvcc --version | tail -2
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF -DGGML_NATIVE=ON
cmake --build "$LLAMA_DIR/build" --config Release -j 18 --target llama-$name llama-cli
ls -la "$LLAMA_DIR/build/bin/" | grep -E "llama-(capture|cli)"
sha256sum "$LLAMA_DIR/build/bin/llama-$name"
echo BUILD_DONE
