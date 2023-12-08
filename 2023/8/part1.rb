#!/usr/bin/env ruby

input = $<.to_enum
directions = input.next.chomp.chars.map { _1 == 'L' ? 0 : 1 }
input.next

children = input.with_object({}) { |line, h|
  id, l, r = /^(\w+) = \((\w+), (\w+)\)/.match(line).captures
  h[id] = [l, r]
}

cur = 'AAA'
cur_dir = directions.to_enum.cycle
steps = 0
loop {
  cur = children[cur][cur_dir.next]
  steps += 1
  break if cur == 'ZZZ'
}
puts steps
