#!/usr/bin/env ruby

def solve(chars, seq)
  unknowns = chars.each_with_index.filter_map { |c, i| i if c == '?' }
  (0...2 ** unknowns.size).count { |m|
    unknowns.each_with_index { |i, j| chars[i] = (m & (1 << j)) == 0 ? '.' : '#' }
    valid?(chars, seq) #.tap { puts chars.join('') if _1}
  }
end

def valid?(chars, seq)
  chars.chunk { _1 }.filter_map { |c, xs| xs.count if c == '#' } == seq
end


puts $<.map { |line|
  pattern, seq = line.chomp.split(' ')
  pattern = pattern.chars
  seq = seq.split(',').map(&:to_i)
  solve(pattern, seq) #.tap { puts "#{_1} #{line}"}
}.sum
