#!/usr/bin/env julia

@views brick_range(state, (a, b)) =
  state[a[1]:b[1], a[2]:b[2], a[3]:b[3]]

bricks_except(self, slice) =
  Iterators.filter(x -> x != 0 && x != self, slice)

function solve_part1(bricks, state)
  count(enumerate(bricks)) do (self, (a, b))
    b[3] == size(state, 3) && return true
    supports = unique(bricks_except(self,
      state[a[1]:b[1], a[2]:b[2], b[3]+1]))
    all(supports) do (brick)
      local (a, b) = bricks[brick]
      !isempty(bricks_except(self, state[a[1]:b[1], a[2]:b[2], a[3]-1]))
    end
  end
end

function solve_part2(bricks, state)
  children = []
  for self in eachindex(bricks)
    (a, b) = bricks[self]
    b[3] == size(state, 3) && return 0
    push!(children, unique(bricks_except(self,
      state[a[1]:b[1], a[2]:b[2], b[3]+1])))
  end
  parents = [Vector{Int}() for _ in eachindex(bricks)]
  for (node, children) in enumerate(children)
    for child in children
      push!(parents[child], node)
    end
  end

  get_falls(falls, node) = begin
    for child in children[node]
      falls[child] = all(parent -> falls[parent], parents[child])
    end
    for child in children[node]
      falls[child] && get_falls(falls, child)
    end
  end

  falls = Array{Bool}(undef, length(bricks))
  sum(eachindex(bricks)) do brick
    fill!(falls, false)
    falls[brick] = true
    get_falls(falls, brick)
    count(falls) - 1
  end
end

bricks = [
  Tuple(
    let (x, y, z) = (parse(Int, i) for i in eachsplit(s, ","))
        [x + 1, y + 1, z]
    end
    for s in eachsplit(line, "~"))
  for line in eachline()
]
sort!(bricks; lt=(a, b) -> a[1][3] < b[1][3])

state = zeros(Int, Tuple(
  maximum(b[d] for (a, b) in bricks)
  for d ∈ 1:3
))
for i in eachindex(bricks)
  brick = bricks[i]
  (a, b) = brick
  brick_range(state, brick) .= i
  while brick[1][3] > 1 && isempty(bricks_except(brick,
    state[a[1]:b[1], a[2]:b[2], a[3]-1]))
    brick_range(state, brick) .= 0
    brick[1][3] -= 1
    brick[2][3] -= 1
    brick_range(state, brick) .= i
  end
end

println("Part 1: ", solve_part1(bricks, state))
println("Part 2: ", solve_part2(bricks, state))
