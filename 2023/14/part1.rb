#!/usr/bin/env ruby

def tilt_north(data)
  data = data.map(&:dup)
  (0...data.size).each { |y|
    (0...data[0].size).each { |x|
      next unless data[y][x] == 'O'
      new_y = (0...y).reverse_each.lazy.
        take_while { data[_1][x] == '.' }.reduce { _2 }
      if new_y
        data[new_y][x] = 'O'
        data[y][x] = '.'
      end
    }
  }
  data
end

def north_load(data)
  (0...data.size).lazy.filter_map { |y|
    (0...data[0].size).lazy.filter_map { |x|
      data.size - y if data[y][x] == 'O'
    }.sum
  }.sum
end

data = $<.readlines(chomp: true)
puts north_load(tilt_north(data))
