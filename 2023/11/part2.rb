#!/usr/bin/env ruby

rows = $<.readlines(chomp: true).map(&:chars)

inflated_y, inflated_x = [rows, rows.transpose].map { |xs|
  xs.reduce([]) { |inflated, col|
    prev = inflated.empty? ? -1 : inflated[-1]
    inflated << prev + (col.all? { _1 == '.' } ? 1000000 : 1)
  }
}

stars = []
(0...rows.size).each { |y|
  (0...rows[0].size).each { |x|
    stars << [inflated_x[x], inflated_y[y]] if rows[y][x] == '#'
  }
}

result = 0
(0...stars.size).each { |i|
  (0...i).each { |j|
    a, b = stars[i], stars[j]
    dist = (a[0] - b[0]).abs + (a[1] - b[1]).abs
    result += dist
  }
}
puts result
