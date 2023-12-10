#!/usr/bin/env ruby

directions = $<.readline(chomp: true).chars.map { _1 == 'L' ? 0 : 1 }
$<.readline
children = $<.each.with_object({}) { |line, h|
  id, left, right = line.scan(/\w+/)
  h[id] = [left, right]
}

cur = 'AAA'
puts directions.cycle.take_while { |dir|
  (cur = children[cur][dir]) != 'ZZZ'
}.count + 1
