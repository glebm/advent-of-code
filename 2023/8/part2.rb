#!/usr/bin/env ruby

input = $<.to_enum
directions = input.next.chomp.chars.map { _1 == 'L' ? 0 : 1 }
input.next

children = input.with_object({}) { |line, h|
  id, l, r = /^(\w+) = \((\w+), (\w+)\)/.match(line).captures
  h[id] = [l, r]
}

# In the dataset, the distance from initial to terminal happens
# to be the same as the distance from that terminal to the next terminal,
# which is always itself.
puts children.keys.select { _1[-1] == 'A' }.map { |cur|
  steps = 0
  cur_dir = directions.to_enum.cycle
  loop {
    cur = children[cur][cur_dir.next]
    steps += 1
    break if cur[-1] == 'Z'
  }
  steps
}.reduce(:lcm)
