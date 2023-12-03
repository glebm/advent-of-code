#!/usr/bin/env ruby
require 'strscan'

LINES = [""]
ARGF.each_line { |line| LINES << ".#{line.chomp}." }
width = LINES[1].size - 2
height = LINES.size - 1
LINES[0] = '.' * (width + 2)
LINES << '.' * (width + 2)

def sym?(c) = /[^0-9.]/.match?(c)

puts (1..height).reduce(0) { |sum, y|
  line = LINES[y]
  scanner = StringScanner.new(line)
  until scanner.eos?
    scanner.skip(/[^\d]+/)
    num = scanner.scan(/\d+/)
    break unless scanner.matched?
    x_beg = scanner.pos - scanner.matched_size
    x_end = scanner.pos
    next unless sym?(line[x_beg - 1]) || sym?(line[x_end]) ||
      [-1, 1].any? { |dy| sym? LINES[y + dy][x_beg - 1..x_end] }
    sum += num.to_i
  end
  sum
}
