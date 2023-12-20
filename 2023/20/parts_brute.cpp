#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <queue>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
constexpr size_t MAX_SIGNALS = 7;
using Mod = std::vector<uint8_t>;
struct Mods {
  std::vector<Mod> all;
  uint8_t flips_end;
  std::vector<uint64_t> conj_masks;
  std::vector<std::string> names;
};

struct Signal {
  bool hi;
  uint8_t src;
};

struct State {
  std::vector<uint8_t /*bool*/> flip;
  std::vector<uint64_t> conj;
  std::array<Signal, MAX_SIGNALS * MAX_SIGNALS> signals;
  bool flipped = false;
  size_t num_signals = 0;

  void signal(bool hi, uint8_t src) { signals[flipped * (signals.size() / 2) + num_signals++] = Signal{hi, src}; }
  
  std::span<Signal> step() {
    auto result = std::span<Signal>(signals.data() + flipped * (signals.size() / 2), num_signals);
    flipped = !flipped;
    num_signals = 0;
    return result;
  }

  State(const Mods &mods)
      : flip(mods.flips_end, 0), conj(mods.all.size() - mods.flips_end, 0) {}
};

void send_hi(const Mods &mods, uint8_t from, uint8_t to, State &state) {
  // std::cout << mods.names[from] << " -high-> " << mods.names[to] << std::endl;
  if (to < mods.flips_end || to >= mods.all.size()) return;
  const Mod &mod = mods.all[to];
  uint64_t &conj = state.conj[to - mods.flips_end];
  conj |= (1UL << from);
  const bool pulse = (conj != mods.conj_masks[to - mods.flips_end]);
  state.signal(pulse, to);
}

void send_lo(const Mods &mods, uint8_t from, uint8_t to, State &state) {
  // std::cout << "broadcaster" << " -low-> " << mods.names[to] << std::endl;
  if (to >= mods.all.size()) return;
  const Mod &mod = mods.all[to];
  if (to < mods.flips_end) {
    const bool pulse = state.flip[to] = !state.flip[to];
    state.signal(pulse, to);
  } else {
    uint64_t &conj = state.conj[to - mods.flips_end];
    conj &= ~(1UL << from);
    state.signal(true, to);
  }
}

uint64_t solve_part1(const Mods &mods) {
  State state{mods};
  uint64_t sent_lo = 0;
  uint64_t sent_hi = 0;
  for (unsigned i = 1; i <= 1000; ++i) {
    ++sent_lo;
    state.signal(false, static_cast<uint8_t>(mods.all.size() - 1));
    while (true) {
      std::span<Signal> cur_signals = state.step();
      if (cur_signals.empty()) break;
      for (const Signal &signal : cur_signals) {
        if (signal.hi) {
          for (const uint8_t dest : mods.all[signal.src]) {
            send_hi(mods, signal.src, dest, state);
          }
          sent_hi += mods.all[signal.src].size();
        } else {
          for (const uint8_t dest : mods.all[signal.src]) {
            send_lo(mods, signal.src, dest, state);
          }
          sent_lo += mods.all[signal.src].size();
        }
      }
    }
    // std::cout << std::endl;
  }
  return sent_hi * sent_lo;
}

uint64_t solve_part2(const Mods &mods, uint8_t rx) {
  State state{mods};
  for (uint64_t i = 1;; ++i) {
    state.signal(false, static_cast<uint8_t>(mods.all.size() - 1));
    while (true) {
      std::span<Signal> cur_signals = state.step();
      if (cur_signals.empty()) break;
      for (const Signal &signal : cur_signals) {
        if (signal.hi) {
          for (const uint8_t dest : mods.all[signal.src]) {
            send_hi(mods, signal.src, dest, state);
          }
        } else {
          for (const uint8_t dest : mods.all[signal.src]) {
            if (dest == rx) return i;
            send_lo(mods, signal.src, dest, state);
          }
        }
      }
    }
    if ((i % 1000000) == 0) {
      std::cout << "Pressed the button " << (i / 1000000) << " million times"
                << std::endl;
    }
  }
  return 0;
}

std::vector<std::string> split(const std::string &s) {
  std::vector<std::string> result;
  size_t begin = 0;
  while (true) {
    const size_t end = s.find(',', begin);
    if (end == std::string::npos) {
      result.push_back(s.substr(begin));
      break;
    }
    result.push_back(s.substr(begin, end - begin));
    begin = end + 2;
  }
  return result;
}

}  // namespace

int main(int argc, char *argv[]) {
  std::vector<std::string> broadcast_s;
  std::vector<std::pair<std::string, std::vector<std::string>>> flip_s;
  std::vector<std::pair<std::string, std::vector<std::string>>> conj_s;

  std::ifstream fin;
  std::istream &in = argc == 2 ? (fin.open(argv[1]), fin) : std::cin;
  // std::istream &in = (fin.open("input"), fin);

  while (true) {
    std::string s;
    std::getline(in, s);
    if (!in) break;
    size_t ws_pos = s.find(' ');
    std::string key = s.substr(0, ws_pos);
    std::string value = s.substr(s.find(' ', ws_pos + 2) + 1);
    if (key == "broadcaster") {
      broadcast_s = split(value);
    } else if (key[0] == '%') {
      flip_s.emplace_back(key.substr(1), split(value));
    } else if (key[0] == '&') {
      conj_s.emplace_back(key.substr(1), split(value));
    }
  }

  Mods mods;
  mods.flips_end = flip_s.size();

  std::unordered_map<std::string, uint8_t> idx;
  for (const auto &arr : {flip_s, conj_s}) {
    for (const auto &[k, v] : arr) idx[k] = idx.size();
  }
  idx["broadcaster"] = idx.size();
  for (const auto &arr : {flip_s, conj_s}) {
    for (const auto &[k, v] : arr) {
      std::vector<uint8_t> &dests = mods.all.emplace_back();
      for (const std::string &d : v) {
        const auto it = idx.find(d);
        if (it == idx.end()) { // sink
          dests.push_back(idx.size());
          idx[d] = idx.size();
        } else {
          dests.push_back(it->second);
        }
      }
    }
  }

  mods.conj_masks.resize(mods.all.size() - mods.flips_end);
  for (size_t conj_i = mods.flips_end; conj_i < mods.all.size(); ++conj_i) {
    uint64_t mask = 0;
    for (size_t src_i = 0; src_i < mods.all.size(); ++src_i) {
      const auto &dests = mods.all[src_i];
      if (std::find(dests.begin(), dests.end(), conj_i) != dests.end()) {
        mask |= (1UL << src_i);
      }
    }
    mods.conj_masks[conj_i - mods.flips_end] = mask;
  }

  auto &broadcast = mods.all.emplace_back();
  for (const std::string &b : broadcast_s) broadcast.push_back(idx[b]);

  mods.names = std::vector<std::string>(idx.size());
  for (const auto &[s, i] : idx) mods.names[i] = (i < mods.flips_end ? "%" : "&") + s;

  std::cout << "Part 1: " << solve_part1(mods) << std::endl;
  if (const auto rx = idx.find("rx"); rx != idx.end()) {
    std::cout << "Part 2: " << solve_part2(mods, rx->second) << std::endl;
  }
  return 0;
}
