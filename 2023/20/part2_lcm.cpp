#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
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

  void signal(bool hi, uint8_t src) {
    signals[flipped * (signals.size() / 2) + num_signals++] = Signal{hi, src};
  }

  std::span<Signal> step() {
    auto result = std::span<Signal>(
        signals.data() + flipped * (signals.size() / 2), num_signals);
    flipped = !flipped;
    num_signals = 0;
    return result;
  }

  State(const Mods &mods)
      : flip(mods.flips_end, 0), conj(mods.all.size() - mods.flips_end, 0) {}
};

void send_hi(const Mods &mods, uint8_t from, uint8_t to, State &state) {
  // std::cout << mods.names[from] << " -high-> " << mods.names[to] <<
  // std::endl;
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

void save_graphviz_dot(const Mods &mods, const State &state,
                       const std::vector<uint8_t> &sinks,
                       const std::vector<uint8_t> &sink_parents, uint8_t target,
                       const std::string &path) {
  constexpr std::string_view kNodeOn =
      " style=\"filled\" fillcolor=\"#b71c1c\"";
  constexpr std::string_view kEdgeHigh = " [color=\"#ef5350\"]";
  std::ofstream out;
  out.open(path);
  const auto name = [&](uint8_t i) {
    return std::string_view(mods.names[i].data() + 1, mods.names[i].size() - 1);
  };
  const auto rank = [&](std::string_view rank, std::vector<uint8_t> nodes) {
    out << "  {\n" << "    rank=\"" << rank << "\"\n";
    for (const uint8_t node : nodes) out << "    " << name(node) << "\n";
    out << "  }\n";
  };

  out << "digraph g {\n";
  for (uint8_t i = 0; i < mods.flips_end; ++i) {
    const bool on = state.flip[i];
    out << "  " << name(i) << " [shape=circle" << (on ? kNodeOn : "") << "]\n";
    out << "  " << name(i) << " -> {";
    for (const uint8_t dest : mods.all[i]) out << " " << name(dest);
    out << " }";
    if (on) out << kEdgeHigh;
    out << "\n";
  }
  for (uint8_t i = mods.flips_end, j = 0; i < mods.all.size(); ++i, ++j) {
    const bool on = mods.conj_masks[j] == state.conj[j];
    out << "  " << name(i) << " [shape=box3d" << (on ? kNodeOn : "") << "]\n";
    out << "  " << name(i) << " -> {";
    for (const uint8_t dest : mods.all[i]) out << " " << name(dest);
    out << " }";
    if (!on) out << kEdgeHigh;
    out << "\n";
  }

  // broadcaster and its destinations
  rank("min", {static_cast<uint8_t>(mods.all.size() - 1)});
  rank("same", mods.all.back());

  rank("sink", sink_parents);
  rank("sink", sinks);
  rank("max", {target});

  out << "}\n";
}

uint64_t solve_part2(const Mods &mods, const std::vector<uint8_t> &sinks,
                     uint8_t rx) {
  // For visualization.
  std::vector<uint8_t> sink_parents;
  for (uint8_t i = mods.flips_end; i < mods.all.size(); ++i) {
    for (const uint8_t sink : sinks) {
      if (std::find(mods.all[i].begin(), mods.all[i].end(), sink) !=
          mods.all[i].end()) {
        sink_parents.push_back(i);
      }
    }
  }

  State state{mods};
  std::vector<uint64_t> cycle_lengths(sinks.size(), 0);
  const auto print_sink_cycles = [&](std::ostream &os) -> std::ostream & {
    for (size_t sink_i = 0; sink_i < sinks.size(); ++sink_i) {
      os << " " << mods.names[sinks[sink_i]] << "=" << cycle_lengths[sink_i];
    }
    return os;
  };
  const auto print_sink_states = [&](std::ostream &os) -> std::ostream & {
    for (size_t sink_i = 0; sink_i < sinks.size(); ++sink_i) {
      const uint8_t sink = sinks[sink_i];
      os << " " << mods.names[sink] << "=";
      const uint64_t st = state.conj[sink - mods.flips_end];
      const uint64_t mask = mods.conj_masks[sink - mods.flips_end];
      os << st << "/" << mask;
    }
    return os;
  };

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
    if (i <= 16384) {
      char path[20];
      sprintf(path, "viz/%06d.dot", static_cast<uint32_t>(i));
      save_graphviz_dot(mods, state, sinks, sink_parents, rx, path);
    }
    bool all = true;
    for (size_t sink_i = 0; sink_i < sinks.size(); ++sink_i) {
      if (cycle_lengths[sink_i] != 0) continue;
      const uint8_t sink = sinks[sink_i];
      if (state.conj[sink - mods.flips_end] !=
          mods.conj_masks[sink - mods.flips_end]) {
        cycle_lengths[sink_i] = i;
      } else {
        all = false;
      }
    }
    if (all) {
      print_sink_cycles(std::cout) << std::endl;
      uint64_t result = 1;
      for (const uint64_t x : cycle_lengths) result = std::lcm(result, x);
      return result;
    }
    if ((i % 1000000) == 0) {
      std::cout << "Pressed the button " << (i / 1000000) << " million times;";
      print_sink_states(std::cout) << " ";
      print_sink_cycles(std::cout) << std::endl;
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
        if (it == idx.end()) {  // sink
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
  for (const auto &[s, i] : idx)
    mods.names[i] = (i < mods.flips_end ? "%" : "&") + s;

  const uint8_t rx = idx.at("rx");
  const uint8_t rx_parent = std::distance(
      mods.all.begin(),
      std::find_if(mods.all.begin(), mods.all.end(), [rx](const auto &dests) {
        return dests.size() == 1 && dests[0] == rx;
      }));
  std::vector<uint8_t> sinks;
  for (size_t i = 0; i < mods.all.size(); ++i) {
    const auto &dests = mods.all[i];
    if (dests.size() == 1 && dests[0] == rx_parent) {
      std::cout << "Sink: " << mods.names[i] << std::endl;
      sinks.push_back(i);
    }
  }

  std::cout << solve_part2(mods, sinks, rx) << std::endl;
  return 0;
}
