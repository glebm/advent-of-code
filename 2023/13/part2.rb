#!/usr/bin/env ruby

def solve(data, ignore = -1)
  (1...data.size).find { |i|
    next if i == ignore
    len = [i, data.size - i].min
    (1..len).all? { data[i - _1] == data[i + _1 - 1] }
  } || 0
end

def smudge(c) = c == '#' ? '.' : '#'

puts loop.map {
  data = $<.lazy.map(&:chomp).take_while { !_1.empty? }.to_a.map(&:chars)
  raise StopIteration if data.empty?
  initial_row = solve(data)
  initial_column = solve(data.transpose)
  row, column = (0...data.size).lazy.filter_map { |y|
    (0...data[0].size).lazy.filter_map { |x|
      data[y][x] = smudge(data[y][x])
      row = solve(data, initial_row)
      column = solve(data.transpose, initial_column)
      data[y][x] = smudge(data[y][x])
      [row, column] if row != 0 || column != 0
    }.first
  }.first
  row * 100 + column
}.sum
