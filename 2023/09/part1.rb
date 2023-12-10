#!/usr/bin/env ruby

lines = $<.map { _1.scan(/\-?\d+/).map(&:to_i) }

def line_diffs(seq) = seq.each_cons(2).map { |a, b| b - a }

def all_diffs(seq)
  result = [seq]
  loop do
    seq = line_diffs(seq)
    result << seq
    break if seq.all?(&:zero?)
  end
  result
end

puts lines.map { |line|
  diffs = all_diffs(line).reverse
  diffs[0] << 0
  (1...diffs.size).each do |i|
    diffs[i] << diffs[i - 1][-1] + diffs[i][-1]
  end
  diffs[-1][-1]
}.reduce(:+)
