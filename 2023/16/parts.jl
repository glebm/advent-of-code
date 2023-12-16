#!/usr/bin/env julia

const Point = @NamedTuple{x::Int, y::Int}
const PosDir = @NamedTuple{pos::Point, dir::Point}
const BeamPointData = Vector{Point}

function run_beams!(
  beams2d::Matrix{BeamPointData}, data::Matrix{Char},
  origin::Point, origin_dir::Point)
  foreach(empty!, beams2d)
  stack = [(pos=origin, dir=origin_dir)]
  push = (point, dir) -> begin
    new_point = (x=point.x + dir.x, y=point.y + dir.y)
    if checkbounds(Bool, beams2d, new_point...)
      push!(stack, (pos=new_point, dir=dir))
    end
  end
  while !isempty(stack)
    (pos::Point, dir::Point) = pop!(stack)
    beams = beams2d[pos.x, pos.y]
    dir in beams && continue
    push!(beams, dir)
    c = data[pos.x, pos.y]
    if c == '.'
      push(pos, dir)
    elseif c == '|'
      if dir.y != 0
        push(pos, dir)
      else
        push(pos, (x=0, y=1))
        push(pos, (x=0, y=-1))
      end
    elseif c == '-'
      if dir.x != 0
        push(pos, dir)
      else
        push(pos, (x=1, y=0))
        push(pos, (x=-1, y=0))
      end
    elseif c == '/'
      push(pos, (x=-dir.y, y=-dir.x))
    elseif c == '\\'
      push(pos, (x=dir.y, y=dir.x))
    end
  end
  beams2d
end

data = stack(Iterators.map(collect, eachline()))
beams = [BeamPointData() for _ in 1:size(data)[1], _ in 1:size(data)[2]]

# Visualization from the puzzle description:
function debug(data, beams2d)
  dir_str((x, y)) =
    x == 0 ? (y == 1 ? 'v' : '^') : (x == 1 ? '>' : '<')
  debug_beams(beams) =
    length(beams) > 1 ? "$(length(beams))" : dir_str(beams[1])
  for (cs, bs) in zip(eachcol(data), eachcol(beams2d))
    println(prod(
      (c != '.' || isempty(b)) ? c : debug_beams(b)
      for (c, b) in zip(cs, bs)))
  end
  nothing
end

count_nonempty(arr) = count(!isempty(c) for c in arr)

run_beams!(beams, data, Point((1, 1)), Point((1, 0)))
# debug(data, beams)
println("Part 1: ", count_nonempty(beams))

w, h = size(data)
result = maximum(Iterators.flatten((
  ((pos=(x=x, y=1), dir=(x=0, y=1)) for x in 1:w),
  ((pos=(x=x, y=h), dir=(x=0, y=-1)) for x in 1:w),
  ((pos=(x=1, y=y), dir=(x=1, y=0)) for y in 1:h),
  ((pos=(x=w, y=y), dir=(x=-1, y=0)) for y in 1:h)
))) do (pos, dir)
  run_beams!(beams, data, pos, dir)
  count_nonempty(beams)
end
println("Part 2: ", result)
