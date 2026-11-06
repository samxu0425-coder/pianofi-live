// Runs NoteDecoder over dumped arrays (onset.f32, offset.f32, mpe.f32,
// velocity.i64, each [frames][88]) in chunks of `chunk` frames and prints the
// notes as CSV. Used by py/check_decoder.py to test the streaming decoder.
//
//   decode_dump DIR [chunk]
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "pf/decoder.h"

template <class T>
std::vector<T> load(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    std::vector<T> v(size_t(f.tellg()) / sizeof(T));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), std::streamsize(v.size() * sizeof(T)));
    return v;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: decode_dump DIR [chunk]\n"); return 2; }
    const std::string d = argv[1];
    const size_t chunk = argc > 2 ? size_t(std::atoi(argv[2])) : 1;
    auto on = load<float>(d + "/onset.f32"), off = load<float>(d + "/offset.f32"), mpe = load<float>(d + "/mpe.f32");
    auto vel = load<int64_t>(d + "/velocity.i64");
    const size_t frames = on.size() / 88;
    pf::NoteDecoder dec;
    for (size_t i = 0; i < frames; i += chunk) {
        const size_t n = std::min(chunk, frames - i);
        dec.add_frames(&on[i * 88], &off[i * 88], &mpe[i * 88], &vel[i * 88], n);
    }
    dec.finish();
    std::printf("pitch,onset,offset,velocity\n");
    for (const auto& n : dec.notes()) std::printf("%d,%.9f,%.9f,%d\n", n.pitch, n.onset, n.offset, n.velocity);
}
