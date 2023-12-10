#!/usr/bin/env ruby
require 'strscan'

LINES = [""]
ARGF.each_line { |line| LINES << ".#{line.chomp}." }
width = LINES[1].size - 2
height = LINES.size - 1
LINES[0] = '.' * (width + 2)
LINES << '.' * (width + 2)
NEIGHBOURS = (-1..1).to_a.product((-1..1).to_a) - [[0, 0]]

def digit?(c) = /\d/.match?(c)

puts (1..height).reduce(0) { |sum, y|
  scanner = StringScanner.new(LINES[y])
  until scanner.eos?
    scanner.scan_until(/\*/)
    break unless scanner.matched?
    x = scanner.pos - 1
    ranges = NEIGHBOURS.filter_map do |dx, dy|
      line = LINES[y + dy]
      x_beg = x + dx
      next nil unless digit?(line[x_beg])
      x_beg -= 1 while digit?(line[x_beg - 1])
      x_end = x_beg + 1
      x_end += 1 while digit?(line[x_end])
      [y + dy, x_beg, x_end]
    end
    nums = ranges.uniq.map { |(y, b, e)| LINES[y][b...e].to_i }
    next unless nums.size == 2
    sum += nums.reduce(:*)
  end
  sum
}
