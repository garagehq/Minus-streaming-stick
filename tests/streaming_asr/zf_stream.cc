// Streaming latency harness for sherpa-onnx RKNN streaming zipformer.
// Feeds a 16 kHz WAV in STEP_MS pieces (as fast as possible), decodes
// whenever ready, and prints one line per newly emitted token (asr) or
// keyword trigger (kws):  fed_s  decode_ms  token  token_ts
//
// usage: zf_stream asr|kws MODELDIR WAV [hotwords_or_keywords_file] [step_ms]
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "sherpa-onnx/c-api/cxx-api.h"

using namespace sherpa_onnx::cxx;
using clk = std::chrono::steady_clock;

static OnlineModelConfig Model(const std::string &dir) {
  OnlineModelConfig m;
  m.transducer.encoder = dir + "/encoder.rknn";
  m.transducer.decoder = dir + "/decoder.rknn";
  m.transducer.joiner = dir + "/joiner.rknn";
  m.tokens = dir + "/tokens.txt";
  m.provider = "rknn";
  m.num_threads = 1;
  return m;
}

int main(int argc, char **argv) {
  if (argc < 4) return 1;
  std::string mode = argv[1], dir = argv[2], extra = argc > 4 ? argv[4] : "";
  int step_ms = argc > 5 ? atoi(argv[5]) : 20;
  Wave w = ReadWave(argv[3]);
  if (w.samples.empty()) return 2;
  int step = w.sample_rate * step_ms / 1000;
  double dec_ms_total = 0, dec_ms_max = 0;
  int n_dec = 0;
  auto timed_decode = [&](auto &r, auto &s) {
    auto t0 = clk::now();
    r.Decode(&s);
    double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    dec_ms_total += ms; n_dec++; if (ms > dec_ms_max) dec_ms_max = ms;
    return ms;
  };

  if (mode == "asr") {
    OnlineRecognizerConfig c;
    c.model_config = Model(dir);
    c.enable_endpoint = true;
    c.rule1_min_trailing_silence = 1.0;
    c.rule2_min_trailing_silence = 0.6;
    c.rule3_min_utterance_length = 15;
    if (!extra.empty()) {
      c.decoding_method = "modified_beam_search";
      c.hotwords_file = extra;
      c.hotwords_score = 2.0;
    }
    auto r = OnlineRecognizer::Create(c);
    auto s = r.CreateStream();
    size_t emitted = 0;
    double seg_off = 0;
    for (size_t i = 0; i < w.samples.size(); i += step) {
      int n = std::min<size_t>(step, w.samples.size() - i);
      s.AcceptWaveform(w.sample_rate, w.samples.data() + i, n);
      double fed = double(i + n) / w.sample_rate;
      double ms = 0;
      while (r.IsReady(&s)) ms += timed_decode(r, s);
      auto res = r.GetResult(&s);
      for (; emitted < res.tokens.size(); emitted++)
        printf("%.3f\t%.1f\t%s\t%.3f\n", fed, ms, res.tokens[emitted].c_str(),
               seg_off + res.timestamps[emitted]);
      if (r.IsEndpoint(&s)) {
        r.Reset(&s);
        emitted = 0;
        seg_off = fed;
      }
    }
  } else {
    KeywordSpotterConfig c;
    c.model_config = Model(dir);
    c.keywords_file = extra;
    c.num_trailing_blanks = 1;
    c.keywords_threshold = argc > 6 ? atof(argv[6]) : 0.25f;
    c.keywords_score = argc > 7 ? atof(argv[7]) : 1.0f;
    auto k = KeywordSpotter::Create(c);
    auto s = k.CreateStream();
    for (size_t i = 0; i < w.samples.size(); i += step) {
      int n = std::min<size_t>(step, w.samples.size() - i);
      s.AcceptWaveform(w.sample_rate, w.samples.data() + i, n);
      double fed = double(i + n) / w.sample_rate;
      while (k.IsReady(&s)) {
        double ms = timed_decode(k, s);
        auto res = k.GetResult(&s);
        if (!res.keyword.empty()) {
          printf("%.3f\t%.1f\t%s\t%.3f\n", fed, ms, res.keyword.c_str(),
                 res.timestamps.empty() ? -1.0 : res.start_time + res.timestamps[0]);
          k.Reset(&s);
        }
      }
    }
  }
  fprintf(stderr, "decodes %d, mean %.1f ms, max %.1f ms, audio %.1f s, total %.1f s\n",
          n_dec, n_dec ? dec_ms_total / n_dec : 0, dec_ms_max,
          double(w.samples.size()) / w.sample_rate, dec_ms_total / 1000);
  return 0;
}
