#!/usr/bin/env ruby

directions = $<.readline(chomp: true).chars.map { _1 == 'L' ? 0 : 1 }
$<.readline
children = $<.each.with_object({}) { |line, h|
  id, left, right = line.scan(/\w+/)
  h[id] = [left, right]
}

# In the dataset, the distance from initial to terminal happens
# to be the same as the distance from that terminal to the next terminal,
# which is always itself.
puts children.keys.filter_map { |cur|
  directions.cycle.lazy.take_while { |dir|
    (cur = children[cur][dir])[-1] != 'Z'
  }.count + 1 if cur[-1] == 'A'
}.reduce(:lcm)
