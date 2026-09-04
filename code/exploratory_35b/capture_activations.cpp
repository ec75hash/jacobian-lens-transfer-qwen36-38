#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-cpp.h"
#include "sampling.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <set>
#include <cstdlib>

namespace fs = std::filesystem;

// ================================================================
// NumPy .npy writer (v1.0, float32, C-contiguous)
// ================================================================
static bool write_npy(const std::string & path, const float * data,
                      const std::vector<int64_t> & shape) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    std::string shape_str = "(";
    for (size_t i = 0; i < shape.size(); i++) {
        shape_str += std::to_string(shape[i]);
        if (i + 1 < shape.size() || shape.size() == 1) shape_str += ",";
    }
    shape_str += ")";

    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape_str + ", }";
    int total = 10 + (int) header.size() + 1;
    int pad = 64 - (total % 64);
    if (pad == 64) pad = 0;
    header += std::string(pad, ' ');
    header += '\n';

    uint16_t header_len = (uint16_t) header.size();
    f.write("\x93NUMPY", 6);
    uint8_t ver[2] = {1, 0};
    f.write((char *) ver, 2);
    f.write((char *) &header_len, 2);
    f.write(header.data(), header_len);

    size_t n_elements = 1;
    for (auto s : shape) n_elements *= s;
    f.write((const char *) data, n_elements * sizeof(float));
    return f.good();
}

static std::string json_escape(const std::string & in) {
    std::string out;
    out.reserve(in.size() + 8);
    for (unsigned char c : in) {
        switch (c) {
            case '\"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back((char) c);
                }
                break;
        }
    }
    return out;
}

// ================================================================
// Tensor accumulator — buffers data across decode calls
// ================================================================
struct tensor_record {
    std::string name;
    std::vector<float> data;
    int64_t dim;
    int64_t n_tokens;
    bool is_router;
};

// ================================================================
// Capture state — shared between main loop and callback
// ================================================================
struct capture_state {
    std::map<std::string, tensor_record> tensors;
    std::set<std::string> discovered_tensor_names;
    std::set<std::string> logged_shape_mismatch;
    std::string output_dir;
    std::string current_prompt_id;
    bool prompt_has_error = false;
    bool list_tensors_mode = false;
    bool routing_only = false;
    bool verbose = false;
    int tensors_seen = 0;
    int tensors_captured = 0;
    std::vector<uint8_t> xfer_buf;
    std::map<int, float> expert_biases;
    int expert_bias_layer = -1;   // -1 = apply at ALL layers (default; bit-identical to old behavior)
    bool dbg_router_name = false; // print raw t->name of each distinct router tensor (Gate-1 verify)
    std::set<std::string> dbg_router_seen;
};

// ================================================================
// Tensor name classification
// ================================================================
static bool is_layer_tensor(const char * name) {
    return strstr(name, "l_out") ||
           strstr(name, "attn_out") ||
           strstr(name, "ffn_out") ||
           strstr(name, "result_output") ||
           strstr(name, "output_norm");
}

static bool is_router_tensor(const char * name) {
    return strstr(name, "ffn_moe_logits") != nullptr;
}

// Extract the layer index N from a router-tensor name containing "ffn_moe_logits-N".
// Returns -1 if the marker or a following digit run isn't found. Parses the digit run
// wherever it sits after the marker — does NOT require N to run to end-of-string — so a
// trailing suffix (e.g. ".weight", "-something") won't break it.
static int router_layer_of(const char * name) {
    const char * marker = "ffn_moe_logits-";
    const char * p = strstr(name, marker);
    if (!p) return -1;
    p += strlen(marker);
    if (*p < '0' || *p > '9') return -1;
    int layer = 0;
    while (*p >= '0' && *p <= '9') {
        layer = layer * 10 + (*p - '0');
        p++;
    }
    return layer;
}

static bool should_capture(const char * name) {
    return is_layer_tensor(name) || is_router_tensor(name);
}

// ================================================================
// Float extraction (handles F32 / F16 / BF16)
// ================================================================
static float get_f32(const uint8_t * data, ggml_type type, const size_t * nb,
                     int64_t i0, int64_t i1) {
    size_t off = (size_t) (i1 * nb[1] + i0 * nb[0]);
    switch (type) {
        case GGML_TYPE_F32:  return *(const float *) &data[off];
        case GGML_TYPE_F16:  return ggml_fp16_to_fp32(*(const ggml_fp16_t *) &data[off]);
        case GGML_TYPE_BF16: return ggml_bf16_to_fp32(*(const ggml_bf16_t *) &data[off]);
        default:             return 0.0f;
    }
}

static void set_f32(uint8_t * data, ggml_type type, const size_t * nb,
                    int64_t i0, int64_t i1, float value) {
    size_t off = (size_t) (i1 * nb[1] + i0 * nb[0]);
    switch (type) {
        case GGML_TYPE_F32:
            *(float *) &data[off] = value;
            break;
        case GGML_TYPE_F16:
            *(ggml_fp16_t *) &data[off] = ggml_fp32_to_fp16(value);
            break;
        case GGML_TYPE_BF16:
            *(ggml_bf16_t *) &data[off] = ggml_fp32_to_bf16(value);
            break;
        default:
            break;
    }
}

static bool parse_expert_bias_spec(const std::string & spec, std::map<int, float> & out) {
    out.clear();
    if (spec.empty()) {
        return true;
    }

    size_t start = 0;
    while (start < spec.size()) {
        size_t end = spec.find(',', start);
        std::string item = spec.substr(start, end == std::string::npos ? std::string::npos : end - start);
        size_t colon = item.find(':');
        if (colon == std::string::npos || colon == 0 || colon + 1 >= item.size()) {
            return false;
        }

        char * expert_tail = nullptr;
        char * bias_tail = nullptr;
        // Keep substrings alive: strtol/strtof set *_tail to point INTO these strings,
        // so they must outlive the checks below (else dangling-pointer UB — broke negatives).
        const std::string expert_str = item.substr(0, colon);
        const std::string bias_str   = item.substr(colon + 1);
        long expert = std::strtol(expert_str.c_str(), &expert_tail, 10);
        float bias = std::strtof(bias_str.c_str(), &bias_tail);
        if (!expert_tail || *expert_tail != '\0' || !bias_tail || *bias_tail != '\0') {
            return false;
        }
        if (expert < 0) {
            return false;
        }
        out[(int) expert] = bias;

        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }

    return true;
}

static void apply_expert_biases(std::vector<float> & f32, int64_t n_tokens, int64_t dim,
                                const std::map<int, float> & expert_biases) {
    if (expert_biases.empty()) {
        return;
    }

    for (const auto & [expert, bias] : expert_biases) {
        if (expert < 0 || expert >= dim) {
            continue;
        }
        for (int64_t tok = 0; tok < n_tokens; tok++) {
            f32[tok * dim + expert] += bias;
        }
    }
}

// ================================================================
// Eval callback — fires for every tensor in the compute graph
// ================================================================
static bool capture_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (capture_state *) user_data;

    if (ask) {
        if (st->list_tensors_mode) return true;
        if (st->routing_only) return is_router_tensor(t->name);
        return should_capture(t->name);
    }

    st->tensors_seen++;

    if (st->dbg_router_name && is_router_tensor(t->name) &&
        st->dbg_router_seen.insert(t->name).second) {
        fprintf(stderr, "[DBG-ROUTER-NAME] raw t->name = '%s'  router_layer_of=%d\n",
                t->name, router_layer_of(t->name));
    }

    if (st->list_tensors_mode) {
        st->discovered_tensor_names.insert(t->name);
        int nd = ggml_n_dims(t);
        LOG("[TENSOR] %-50s  type=%-6s  dims=[", t->name, ggml_type_name(t->type));
        for (int d = 0; d < nd; d++) LOG("%lld%s", (long long) t->ne[d], d < nd - 1 ? ", " : "");
        LOG("]  %s%s\n",
            is_layer_tensor(t->name) ? "LAYER" : "",
            is_router_tensor(t->name) ? "ROUTER" : "");
        return true;
    }

    if (ggml_is_quantized(t->type)) return true;

    const bool on_host = ggml_backend_buffer_is_host(t->buffer);
    size_t nbytes = ggml_nbytes(t);
    if (!on_host) {
        st->xfer_buf.resize(nbytes);
        ggml_backend_tensor_get(t, st->xfer_buf.data(), 0, nbytes);
    }
    uint8_t * src = on_host ? (uint8_t *) t->data : st->xfer_buf.data();

    int64_t dim = t->ne[0];
    int64_t n_tokens = (ggml_n_dims(t) > 1) ? t->ne[1] : 1;

    size_t chunk = (size_t) (n_tokens * dim);
    std::vector<float> f32(chunk);
    for (int64_t tok = 0; tok < n_tokens; tok++) {
        for (int64_t d = 0; d < dim; d++) {
            f32[tok * dim + d] = get_f32(src, t->type, t->nb, d, tok);
        }
    }

    if (is_router_tensor(t->name) && !st->expert_biases.empty() &&
        (st->expert_bias_layer < 0 || router_layer_of(t->name) == st->expert_bias_layer)) {
        apply_expert_biases(f32, n_tokens, dim, st->expert_biases);
        for (int64_t tok = 0; tok < n_tokens; tok++) {
            for (int64_t d = 0; d < dim; d++) {
                set_f32(src, t->type, t->nb, d, tok, f32[tok * dim + d]);
            }
        }
        if (!on_host) {
            ggml_backend_tensor_set(t, src, 0, nbytes);
        }
    }

    std::string name(t->name);
    auto it = st->tensors.find(name);
    if (it != st->tensors.end()) {
        auto & rec = it->second;
        if (rec.dim != dim) {
            if (st->logged_shape_mismatch.insert(name).second) {
                LOG_ERR("  tensor dim mismatch for %s: expected %lld got %lld\n",
                        name.c_str(), (long long) rec.dim, (long long) dim);
            }
            st->prompt_has_error = true;
            return true;
        }
        rec.data.insert(rec.data.end(), f32.begin(), f32.end());
        rec.n_tokens += n_tokens;
    } else {
        tensor_record rec;
        rec.name = name;
        rec.data = std::move(f32);
        rec.dim = dim;
        rec.n_tokens = n_tokens;
        rec.is_router = is_router_tensor(t->name);
        st->tensors[name] = std::move(rec);
    }

    st->tensors_captured++;
    return true;
}

// ================================================================
// Save all captured tensors for one prompt
// ================================================================
struct generated_token_info {
    int32_t token_id;
    std::string piece;
};

struct prompt_token_info {
    int32_t index;
    int32_t token_id;
    std::string piece;
    int32_t start_char;
    int32_t end_char;
};

static std::vector<prompt_token_info> build_prompt_token_info(
        const llama_vocab * vocab,
        const std::vector<llama_token> & tokens,
        const std::string & prompt_text) {
    std::vector<prompt_token_info> out;
    out.reserve(tokens.size());

    size_t cursor = 0;
    for (size_t i = 0; i < tokens.size(); i++) {
        char buf[512];
        int n = llama_token_to_piece(vocab, tokens[i], buf, sizeof(buf), 0, true);
        std::string piece(buf, n > 0 ? n : 0);

        int32_t start_char = -1;
        int32_t end_char = -1;

        if (!piece.empty()) {
            const bool matches_prompt =
                cursor + piece.size() <= prompt_text.size() &&
                prompt_text.compare(cursor, piece.size(), piece) == 0;
            if (matches_prompt) {
                start_char = (int32_t) cursor;
                cursor += piece.size();
                end_char = (int32_t) cursor;
            } else if (cursor > 0) {
                LOG_ERR("  prompt token span mismatch at token %zu ('%s') for %s\n",
                        i, piece.c_str(), prompt_text.substr(cursor, 64).c_str());
            }
        }

        out.push_back({
            (int32_t) i,
            (int32_t) tokens[i],
            piece,
            start_char,
            end_char,
        });
    }

    if (cursor != prompt_text.size()) {
        LOG_ERR("  prompt token spans consumed %zu/%zu chars; metadata boundaries may be unavailable\n",
                cursor, prompt_text.size());
    }

    return out;
}

static void save_prompt(capture_state & st, const std::string & prompt_text,
                        const std::string & generated_text,
                        const std::vector<prompt_token_info> & prompt_tokens,
                        const std::vector<generated_token_info> & generated_tokens,
                        int n_tok_prompt, int n_tok_gen, double ms) {
    fs::path base = fs::path(st.output_dir);
    int n_layer = 0, n_router = 0;

    for (auto & [name, rec] : st.tensors) {
        fs::path dir = base / (rec.is_router ? "router" : "layers");
        fs::create_directories(dir);

        std::string fname = name;
        for (char & c : fname) {
            if (c == '/' || c == ' ' || c == '.') c = '_';
        }

        std::vector<int64_t> shape = {rec.n_tokens, rec.dim};
        if (write_npy((dir / (fname + ".npy")).string(), rec.data.data(), shape)) {
            if (rec.is_router) n_router++; else n_layer++;
        }
    }

    {
        std::ofstream f((base / "metadata.txt").string());
        std::string esc = prompt_text;
        std::replace(esc.begin(), esc.end(), '\n', ' ');
        f << "prompt_id=" << st.current_prompt_id << "\n"
          << "prompt=" << esc << "\n"
          << "n_tokens_prompt=" << n_tok_prompt << "\n"
          << "n_tokens_generated=" << n_tok_gen << "\n"
          << "n_layer_tensors=" << n_layer << "\n"
          << "n_router_tensors=" << n_router << "\n"
          << "elapsed_ms=" << (int) ms << "\n";
    }

    {
        std::ofstream f((base / "prompt_tokens.json").string());
        f << "[\n";
        for (size_t i = 0; i < prompt_tokens.size(); i++) {
            const auto & tok = prompt_tokens[i];
            f << "  {\"index\": " << tok.index
              << ", \"token_id\": " << tok.token_id
              << ", \"piece\": \"" << json_escape(tok.piece) << "\""
              << ", \"start_char\": ";
            if (tok.start_char >= 0) {
                f << tok.start_char;
            } else {
                f << "null";
            }
            f << ", \"end_char\": ";
            if (tok.end_char >= 0) {
                f << tok.end_char;
            } else {
                f << "null";
            }
            f << "}";
            if (i + 1 < prompt_tokens.size()) {
                f << ",";
            }
            f << "\n";
        }
        f << "]\n";
    }

    if (!generated_text.empty()) {
        std::ofstream f((base / "generated_text.txt").string());
        f << generated_text;
    }

    {
        std::ofstream f((base / "generated_tokens.json").string());
        f << "[\n";
        for (size_t i = 0; i < generated_tokens.size(); i++) {
            const auto & tok = generated_tokens[i];
            f << "  {\"step\": " << i
              << ", \"token_id\": " << tok.token_id
              << ", \"piece\": \"" << json_escape(tok.piece) << "\"}";
            if (i + 1 < generated_tokens.size()) {
                f << ",";
            }
            f << "\n";
        }
        f << "]\n";
    }

    LOG_INF("  saved %d layer + %d router tensors → %s\n",
            n_layer, n_router, st.output_dir.c_str());
}

// ================================================================
// Parse tab-separated prompt file
// ================================================================
struct prompt_entry {
    std::string id;
    std::string text;
};

static std::string unescape_prompt(const std::string & raw) {
    std::string out;
    out.reserve(raw.size());
    for (size_t i = 0; i < raw.size(); i++) {
        if (raw[i] == '\\' && i + 1 < raw.size()) {
            char next = raw[i + 1];
            if (next == 'n') { out.push_back('\n'); i++; continue; }
            if (next == 't') { out.push_back('\t'); i++; continue; }
            if (next == '\\') { out.push_back('\\'); i++; continue; }
        }
        out.push_back(raw[i]);
    }
    return out;
}

static std::vector<prompt_entry> parse_prompt_file(const std::string & path) {
    std::vector<prompt_entry> out;
    std::ifstream f(path);
    if (!f.is_open()) {
        LOG_ERR("failed to open prompt file: %s\n", path.c_str());
        return out;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) continue;
        auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        out.push_back({line.substr(0, tab), unescape_prompt(line.substr(tab + 1))});
    }
    return out;
}

static std::string sanitize_prompt_id(const std::string & raw, size_t fallback_idx) {
    if (raw.empty()) {
        return "prompt_" + std::to_string(fallback_idx + 1);
    }

    std::string out;
    out.reserve(raw.size());
    for (char c : raw) {
        const bool ok = (c >= 'a' && c <= 'z') ||
                        (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') ||
                        c == '_' || c == '-';
        out.push_back(ok ? c : '_');
    }

    if (out.empty()) {
        out = "prompt_" + std::to_string(fallback_idx + 1);
    }
    return out;
}

// ================================================================
// Pre-parse custom CLI flags, strip them from argv
// ================================================================
struct custom_args {
    std::string prompt_file;
    std::string output_dir = "./activations";
    bool list_tensors = false;
    bool routing_only = false;
    bool no_stream = false;
    std::string expert_bias_spec;
    int expert_bias_layer = -1;   // -1 = all layers (default)
    bool dbg_router_name = false;
    bool stop_at_eog = false;     // stop generation at ANY end-of-generation token (e.g. <|im_end|>), not just primary EOS
};

static custom_args strip_custom_args(int & argc, char **& argv) {
    custom_args ca;
    std::vector<char *> keep;
    keep.push_back(argv[0]);

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--prompt-file" && i + 1 < argc) {
            ca.prompt_file = argv[++i];
        } else if ((a == "-o" || a == "--output-dir") && i + 1 < argc) {
            ca.output_dir = argv[++i];
        } else if (a == "--list-tensors") {
            ca.list_tensors = true;
        } else if (a == "--routing-only") {
            ca.routing_only = true;
        } else if (a == "--no-stream") {
            ca.no_stream = true;
        } else if (a == "--expert-bias" && i + 1 < argc) {
            ca.expert_bias_spec = argv[++i];
        } else if (a == "--expert-bias-layer" && i + 1 < argc) {
            ca.expert_bias_layer = std::atoi(argv[++i]);
        } else if (a == "--dbg-router-name") {
            ca.dbg_router_name = true;
        } else if (a == "--stop-at-eog") {
            ca.stop_at_eog = true;
        } else {
            keep.push_back(argv[i]);
        }
    }

    argc = (int) keep.size();
    for (int i = 0; i < argc; i++) argv[i] = keep[i];
    return ca;
}

// ================================================================
// Main
// ================================================================
int main(int argc, char ** argv) {
    custom_args ca = strip_custom_args(argc, argv);

    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    common_init();

    capture_state state;
    state.list_tensors_mode = ca.list_tensors;
    state.routing_only = ca.routing_only;
    state.verbose = (params.verbosity > 0);
    state.expert_bias_layer = ca.expert_bias_layer;
    state.dbg_router_name = ca.dbg_router_name;
    if (!parse_expert_bias_spec(ca.expert_bias_spec, state.expert_biases)) {
        LOG_ERR("Failed to parse --expert-bias '%s' (expected expert:bias[,expert:bias...])\n",
                ca.expert_bias_spec.c_str());
        return 1;
    }

    params.cb_eval = capture_cb;
    params.cb_eval_user_data = &state;
    params.warmup = false;

    LOG_INF("=== Consciousness Metrics — Activation Capture ===\n");
    LOG_INF("Output dir : %s\n", ca.output_dir.c_str());
    LOG_INF("n_predict  : %d\n", params.n_predict);
    LOG_INF("Text output: %s\n", ca.no_stream ? "file only (--no-stream)" : "stdout + file");
    if (ca.list_tensors) LOG_INF("Mode       : --list-tensors (discovery)\n");
    if (ca.routing_only) LOG_INF("Mode       : --routing-only (router/gate tensors only)\n");
    if (!state.expert_biases.empty()) {
        LOG_INF("Expert bias: %s\n", ca.expert_bias_spec.c_str());
        LOG_INF("Bias layer : %s\n",
                ca.expert_bias_layer < 0 ? "ALL layers" : std::to_string(ca.expert_bias_layer).c_str());
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx = llama_init->context();
    if (!model || !ctx) { LOG_ERR("model/context init failed\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    const int32_t eos_id = llama_vocab_eos(vocab);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    common_sampler * smpl = nullptr;

    if (params.n_predict > 0 && !ca.list_tensors) {
        smpl = common_sampler_init(model, params.sampling);
        if (!smpl) {
            LOG_ERR("failed to initialize sampler\n");
            return 1;
        }
        LOG_INF("Sampler    : %s\n", common_sampler_print(smpl).c_str());
    }

    LOG_INF("EOS token id: %d\n", eos_id);
    LOG_INF("EOG tokens in vocab:\n");
    for (int i = 0; i < n_vocab; i++) {
        if (llama_vocab_is_eog(vocab, i)) {
            char buf[256];
            int n = llama_token_to_piece(vocab, i, buf, sizeof(buf), 0, true);
            std::string tok_str(buf, n > 0 ? n : 0);
            LOG_INF("  token %d = '%s'%s\n", i, tok_str.c_str(),
                    i == eos_id ? " [PRIMARY EOS - will stop generation]" : " [EOG - will NOT stop generation]");
        }
    }

    LOG_INF("%s\n\n", common_params_get_system_info(params).c_str());

    std::vector<prompt_entry> prompts;
    if (!ca.prompt_file.empty()) {
        prompts = parse_prompt_file(ca.prompt_file);
        if (prompts.empty()) {
            LOG_ERR("No valid prompts loaded from --prompt-file %s\n", ca.prompt_file.c_str());
            return 1;
        }
    } else if (!params.prompt.empty()) {
        prompts.push_back({"single", params.prompt});
    } else {
        LOG_ERR("No prompts.  Use --prompt-file <tsv> or -p <text>\n");
        return 1;
    }
    LOG_INF("Loaded %zu prompts\n\n", prompts.size());

    std::set<std::string> used_prompt_dirs;
    for (size_t pi = 0; pi < prompts.size(); pi++) {
        const auto & pe = prompts[pi];
        const std::string safe_id_base = sanitize_prompt_id(pe.id, pi);
        std::string safe_id = safe_id_base;
        int suffix = 2;
        while (!used_prompt_dirs.insert(safe_id).second) {
            safe_id = safe_id_base + "_" + std::to_string(suffix++);
        }
        if (safe_id_base != pe.id || safe_id != safe_id_base) {
            LOG_INF("  sanitized prompt id '%s' -> '%s'\n", pe.id.c_str(), safe_id.c_str());
        }

        LOG_INF("[%zu/%zu] %s : %.70s%s\n",
                pi + 1, prompts.size(), pe.id.c_str(),
                pe.text.c_str(), pe.text.size() > 70 ? "..." : "");

        state.tensors.clear();
        state.discovered_tensor_names.clear();
        state.logged_shape_mismatch.clear();
        state.tensors_seen = 0;
        state.tensors_captured = 0;
        state.prompt_has_error = false;
        state.current_prompt_id = pe.id;
        state.output_dir = (fs::path(ca.output_dir) / safe_id).string();
        fs::create_directories(state.output_dir);

        llama_memory_clear(llama_get_memory(ctx), true);
        if (smpl) {
            common_sampler_reset(smpl);
        }

        std::vector<llama_token> tokens = common_tokenize(ctx, pe.text, add_bos, /*parse_special=*/true);  // chat-template special tokens must be single ids
        int n_prompt = (int) tokens.size();
        if (tokens.empty()) {
            LOG_ERR("  empty tokenization — skipping\n");
            continue;
        }
        std::vector<prompt_token_info> prompt_tokens = build_prompt_token_info(vocab, tokens, pe.text);

        auto t0 = std::chrono::high_resolution_clock::now();

        const int prefill_batch = params.n_batch > 0 ? params.n_batch : 512;
        for (int pos = 0; pos < n_prompt; ) {
            int n_cur = std::min(prefill_batch, n_prompt - pos);
            if (llama_decode(ctx, llama_batch_get_one(tokens.data() + pos, n_cur))) {
                LOG_ERR("  prefill decode failed at token %d/%d — skipping\n", pos, n_prompt);
                state.prompt_has_error = true;
                break;
            }
            if (state.prompt_has_error) {
                break;
            }
            pos += n_cur;
        }

        if (state.prompt_has_error) {
            LOG_ERR("  prompt capture failed during prefill; removing partial output\n");
            fs::remove_all(state.output_dir);
            continue;
        }

        int n_gen = 0;
        std::string gen_text;
        std::vector<generated_token_info> generated_tokens;

        if (params.n_predict > 0 && !ca.list_tensors) {
            for (int g = 0; g < params.n_predict; g++) {
                llama_token next = common_sampler_sample(smpl, ctx, -1);
                if (next == eos_id) break;
                if (ca.stop_at_eog && llama_vocab_is_eog(vocab, next)) break;

                n_gen++;
                {
                    char buf[256];
                    int n = llama_token_to_piece(vocab, next, buf, sizeof(buf), 0, true);
                    std::string piece = n > 0 ? std::string(buf, n) : std::string();
                    generated_tokens.push_back({next, piece});
                    if (n > 0) {
                        gen_text.append(piece);
                        if (!ca.no_stream) {
                            fwrite(buf, 1, n, stdout);
                            fflush(stdout);
                        }
                    }
                }
                common_sampler_accept(smpl, next, true);

                if (llama_decode(ctx, llama_batch_get_one(&next, 1))) {
                    LOG_ERR("  gen decode failed at token %d\n", g);
                    state.prompt_has_error = true;
                    break;
                }
                if (state.prompt_has_error) {
                    LOG_ERR("  prompt capture failed during generation; aborting prompt\n");
                    break;
                }
            }
        }

        if (!ca.no_stream && n_gen > 0) {
            fprintf(stdout, "\n");
            fflush(stdout);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        LOG_INF("  tokens: %d prompt + %d gen | tensors: %d seen, %d captured, %zu unique | %.0f ms\n",
                n_prompt, n_gen,
                state.tensors_seen, state.tensors_captured, state.tensors.size(),
                ms);

        if (ca.list_tensors) {
            LOG_INF("\nDiscovered %zu unique tensor names.  Exiting.\n", state.discovered_tensor_names.size());
            break;
        }

        if (state.prompt_has_error) {
            LOG_ERR("  prompt capture failed; removing partial output\n");
            fs::remove_all(state.output_dir);
            continue;
        }

        save_prompt(state, pe.text, gen_text, prompt_tokens, generated_tokens, n_prompt, n_gen, ms);
    }

    common_sampler_free(smpl);

    LOG_INF("\n=== Capture Complete ===\n");
    LOG_INF("Prompts: %zu  |  Output: %s\n", prompts.size(), ca.output_dir.c_str());
    llama_perf_context_print(ctx);
    llama_backend_free();
    return 0;
}
