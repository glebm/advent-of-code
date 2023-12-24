#!/usr/bin/env -S julia --threads=auto -O3 --check-bounds=no

using LinearAlgebra
using LinearSolve
using StaticArrays

const DEBUG_LOG::Bool = true
const V_RANGE::Int = 1000
const T_RANGE::Int = 2000000

@inbounds @views function intersect2d(a, b, minxy, maxxy)
  a0, b0 = a[1][1:2], b[1][1:2]
  va, vb = a[2][1:2], b[2][1:2]
  tb, ta = try
    inv(hcat(vb, -va)) * (a0 - b0)
  catch e
    !isa(e, SingularException) && throw(e)
    return false
  end
  pos = a0 + ta .* va
  tb >= 0 && ta >= 0 && all(minxy ≤ p ≤ maxxy for p ∈ pos)
end

function solve_part1(stones, minxy, maxxy)
  result = 0
  for i in eachindex(stones), j in 1:i-1
    if intersect2d(stones[j], stones[i], minxy, maxxy)
      result += 1
    end
  end
  result
end

@inline @views function try_solution_xyz(stones, p::SVector{3,Int}, v::SVector{3,Int})::Bool
  px, py, pz = p
  vx, vy, vz = v
  for stone in stones
    spx, spy, spz = stone[1]
    svx, svy, svz = stone[2]
    (px - spx) * (svy - vy) != (svx - vx) * (py - spy) && return false
    (pz - spz) * (svy - vy) != (svz - vz) * (py - spy) && return false
  end
  true
end

@inline @views function try_solution_xy(stones, px::Int, py::Int, vx::Int, vy::Int)::Bool
  for st in stones
    if (px - st[1][1]) * (st[2][2] - vy) != (st[2][1] - vx) * (py - st[1][2])
      return false
    end
  end
  true
end

@inline @views function solve_given_velocity_xyz(stones, v)::Union{Nothing,SVector{3,Int}}
  for (sp, sv) in stones, t ∈ 1:T_RANGE
    p::SVector{3,Int} = sp + t * (sv - v)
    if try_solution_xyz(stones, p, v)
      DEBUG_LOG && println("3d p=$p v=$v")
      return p
    end
  end
end

@inline @views function solve_given_velocity_xy(stones, vx::Int, vy::Int)::Union{Nothing,NTuple{2,Int}}
  for st in stones, t ∈ 1:T_RANGE
    # p::SVector{2, Int} = sp[1:2] + t * (sv[1:2] - v)
    px::Int = st[1][1] + t * (st[2][1] - vx)
    py::Int = st[1][2] + t * (st[2][2] - vy)
    (st[2][1] == vx && st[1][1] != px || st[2][2] == vy && st[1][2] != py) && return nothing
    if try_solution_xy(stones, px, py, vx, vy)
      DEBUG_LOG && println("2d p=($px, $py) v=($vx, $vy)")
      return (px, py)
    end
  end
end

const Result2D = Tuple{NTuple{2,Int},NTuple{2,Int}}

@views function find_2d_solutions(stones, vs)::Vector{Result2D}
  result::Vector{Result2D} = []
  for (x, y) in vs
    p = solve_given_velocity_xy(stones, x, y)
    if !isnothing(p)
      push!(result, (p, (x, y)))
    end
  end
  result
end

@views function solve_part2(stones)
  stones_minmax_v::Vector{Tuple{Int,Int}} = [extrema(s -> s[2][i], stones) for i in 1:3]
  ranges::Vector{UnitRange{Int}} = [min-V_RANGE:max+V_RANGE for (min, max) in stones_minmax_v]

  total_size_2d::Int = prod(length(r) for r in ranges[1:2])
  chunk_size_2d::Int = max(1, length(ranges[1]) ÷ Threads.nthreads())
  DEBUG_LOG && println("Trying ", total_size_2d, " 𝑣 values (2D) with ",
    Threads.nthreads(), " threads chunk_size=$chunk_size_2d⋅$(length(ranges[2])) = ",
    chunk_size_2d * length(ranges[2]))
  tasks = map(Iterators.partition(ranges[1], chunk_size_2d)) do x_range
    Threads.@spawn find_2d_solutions(stones, Iterators.product(x_range, ranges[2]))
  end
  results::Vector{Vector{Result2D}} = fetch.(tasks)
  ans2ds::Vector{Result2D} = reduce(vcat, results)
  DEBUG_LOG && println("2D answers: ", ans2ds)

  total_size = length(ans2ds) * length(ranges[3])
  println("Trying ", total_size, " 𝑣 values")
  for (i, (p_xy::NTuple{2,Int}, v_xy::NTuple{2,Int})) in enumerate(ans2ds), v_z in ranges[3]
    pos = solve_given_velocity_xyz(stones, @SVector [v_xy[1], v_xy[2], v_z])
    !isnothing(pos) && return sum(pos)
    if i % 10000 == 0
      println("Done ", i, " iterations (", (i * 100) ÷ total_size, "%)")
    end
  end
end

parse_ints(s) = SVector{3,Int}(
  map(v -> parse(Int, v),
    match(r" *([-\d]+), *([-\d]+), *([-\d]+)", s).captures))

hailstones = [
  Tuple(parse_ints(s) for s in eachsplit(line, " @ "))
  for line in eachline()
]

println("Part 1: ", solve_part1(hailstones, 200000000000000, 400000000000000))
println("Part 2: ", solve_part2(hailstones))
