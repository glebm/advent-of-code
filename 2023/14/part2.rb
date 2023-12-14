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

def transpose(data) = data.map(&:chars).transpose.map(&:join)
def reverse(data) = data.reverse

def tilt_south(data) = reverse tilt_north reverse data
def tilt_west(data) = transpose tilt_north transpose data
def tilt_east(data) = transpose reverse tilt_north reverse transpose data
def cycle(data) = tilt_east tilt_south tilt_west tilt_north data

def cycles(data, n)
  seq = [data]
  cycle_begin = 0
  loop do
    data = cycle(data)
    idx = seq.index(data)
    if !idx.nil?
      cycle_begin = idx
      break
    end
    seq << data
  end
  return seq[n] if n < cycle_begin
  seq[cycle_begin + ((n - cycle_begin) % (seq.size - cycle_begin))]
end

data = $<.readlines(chomp: true)
data = cycles(data, 1000000000)
puts north_load(data)
