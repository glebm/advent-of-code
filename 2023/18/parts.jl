#!/usr/bin/env julia

const DIRS = Dict('U' => CartesianIndex(0, -1), 'R' => CartesianIndex(1, 0),
  'D' => CartesianIndex(0, 1), 'L' => CartesianIndex(-1, 0))

function solve(data)
  area = 0
  p = CartesianIndex(1, 1)
  perimeter = 0
  for (dir, dist) in data
    np = p + DIRS[dir] * dist
    perimeter += dist
    area += np[1] * p[2] - np[2] * p[1]
    p = np
  end
  (abs(area) + perimeter) ÷ 2 + 1
end

function parse1(line)
  dir, dist = match(r"^(.) (\d+)", line)
  dir[1], parse(Int, dist)
end

const DIR_LETTER = Dict('0' => 'R', '1' => 'D', '2' => 'L', '3' => 'U')
function parse2(line)
  hex_str, = match(r".*?\(#(.*?)\)$", line)
  DIR_LETTER[hex_str[6]], parse(Int, hex_str[1:5]; base=16)
end

data = readlines()
println("Part 1: ", solve(Iterators.map(parse1, data)))
println("Part 2: ", solve(Iterators.map(parse2, data)))
