#!/usr/bin/env ruby

def solve(data)
  (1...data.size).find { |i|
    len = [i, data.size - i].min
    (1..len).all? { data[i - _1] == data[i + _1 - 1] }
  } || 0
end

puts loop.map {
  data = $<.lazy.map(&:chomp).take_while { !_1.empty? }.to_a.map(&:chars)
  raise StopIteration if data.empty?
  row = solve(data)
  column = solve(data.transpose)
  row * 100 + column
}.sum
