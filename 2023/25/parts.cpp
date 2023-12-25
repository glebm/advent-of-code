#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Edges = std::unordered_map<int, std::vector<int>>;

void dfs(const Edges &edges, size_t start, std::vector<uint8_t> &visited) {
  std::vector<size_t> stack;
  stack.push_back(start);
  visited[start] = 1;
  while (!stack.empty()) {
    const size_t u = stack.back();
    stack.pop_back();
    for (const size_t v : edges.at(u)) {
      if (!visited[v]) {
        visited[v] = 1;
        stack.push_back(v);
      }
    }
  }
}

uint64_t solve(const Edges &edges, const std::vector<std::string> &id2name) {
  std::vector<uint8_t> visited(edges.size(), 0);
  size_t prev_total_visited = 0;
  uint64_t result = 1;
  for (size_t start = 0; start < visited.size(); ++start) {
    if (visited[start] == 1) continue;
    dfs(edges, start, visited);
    const uint64_t total_visited =
        std::count(visited.begin(), visited.end(), 1);
    const uint64_t new_visited = total_visited - prev_total_visited;
    result *= new_visited;
    std::cout << "group: " << new_visited << std::endl;
    prev_total_visited = total_visited;
  }
  return result;
}

std::vector<std::string> split(const std::string &s, char sep) {
  std::vector<std::string> result;
  size_t begin = 0;
  while (true) {
    const size_t end = s.find(sep, begin);
    if (end == std::string::npos) {
      result.push_back(s.substr(begin));
      break;
    }
    result.push_back(s.substr(begin, end - begin));
    begin = end + 1;
  }
  return result;
}

}  // namespace

int main(int argc, char *argv[]) {
  std::vector<std::pair<std::string, std::vector<std::string>>> edges_str;
  std::ifstream fin;
  std::istream &in = argc == 2 ? (fin.open(argv[1]), fin) : std::cin;
  // std::istream &in = (fin.open("input"), fin);
  while (true) {
    std::string s;
    std::getline(in, s);
    if (!in) break;
    size_t col_pos = s.find(':');
    edges_str.emplace_back(s.substr(0, col_pos),
                           split(s.substr(col_pos + 2), ' '));
  }
  size_t n = 0;
  std::unordered_map<std::string, int> ids;

  // Render this input with graphviz neato layout to easily find
  // the edges to remove;
  const std::array<std::pair<std::string, std::string>, 3> exclude{
      std::make_pair<std::string, std::string>("fvh", "fch"),
      std::make_pair<std::string, std::string>("nvg", "vfj"),
      std::make_pair<std::string, std::string>("sqh", "jbz")};

  for (const auto &[k, vs] : edges_str) {
    if (ids.find(k) == ids.end()) ids[k] = n++;
    for (const std::string &v : vs) {
      if (ids.find(v) == ids.end()) ids[v] = n++;
    }
  }
  Edges edges;
  for (const auto &[k, vs] : edges_str) {
    for (const std::string &v : vs) {
      if (std::find(exclude.begin(), exclude.end(),
                    std::pair<std::string, std::string>(k, v)) ==
              exclude.end() &&
          std::find(exclude.begin(), exclude.end(),
                    std::pair<std::string, std::string>(v, k)) ==
              exclude.end()) {
        edges[ids[k]].push_back(ids[v]);
        edges[ids[v]].push_back(ids[k]);
      }
    }
  }
  std::vector<std::string> id2name(edges.size());
  for (const auto &[k, v] : ids) id2name[v] = k;
  std::cout << solve(edges, id2name) << std::endl;
  return 0;
}
