#!/usr/bin/env ruby
require 'strscan'

LINES = [""]
ARGF.each_line { |line| LINES << ".#{line.chomp}." }
width = LINES[1].size - 2
height = LINES.size - 1
LINES[0] = '.' * (width + 2)
LINES << '.' * (width + 2)
NEIGHBOURS = ([1, 0, -1].repeated_combination(2).to_a - [0, 0]).
  flat_map { |d| d.permutation(2).to_a.uniq }

def digit?(c) = /\d/.match?(c)

puts (1..height).reduce(0) { |sum, y|
  line = LINES[y]
  scanner = StringScanner.new(line)
  until scanner.eos?
    scanner.scan_until(/\*/)
    break unless scanner.matched?
    x = scanner.pos - 1
    ranges = NEIGHBOURS.map do |dx, dy|
      line = LINES[y + dy]
      x_beg = x + dx
      next nil unless digit?(line[x_beg])
      x_beg -= 1 while digit?(line[x_beg - 1])
      x_end = x_beg + 1
      x_end += 1 while digit?(line[x_end])
      [y + dy, x_beg, x_end]
    end
    nums = ranges.reject(&:nil?).uniq.map do |(y, b, e)|
      LINES[y][b...e].to_i
    end
    next unless nums.size == 2
    sum += nums.reduce(:*)
  end
  sum
}
