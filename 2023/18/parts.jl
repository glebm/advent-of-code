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
  inner = (abs(area) - perimeter) // 2 + 1
  Int(inner + perimeter)
end

function parse_line(line)
  dir, dist, col = match(r"(.) (\d+) \(#(.*?)\)", line)
  dir[1], parse(Int, dist), col
end

const DIR_LETTER = Dict('0' => 'R', '1' => 'D', '2' => 'L', '3' => 'U')
parse_hex(hex_str) = DIR_LETTER[hex_str[6]], parse(Int, hex_str[1:5]; base=16)

data = map(parse_line, eachline())
println("Part 1: ", solve(((dir, dist) for (dir, dist, _) in data)))
println("Part 2: ", solve((parse_hex(col) for (_, _, col) in data)))
