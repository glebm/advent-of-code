#!/usr/bin/env julia

const NEIGHBOURS = (CartesianIndex(0, -1), CartesianIndex(1, 0),
  CartesianIndex(0, 1), CartesianIndex(-1, 0))

const SLOPES = Dict(
  '^' => CartesianIndex(0, -1), '>' => CartesianIndex(1, 0),
  'v' => CartesianIndex(0, 1), '<' => CartesianIndex(-1, 0))

function visit_part1(data, path::Vector{CartesianIndex{2}}, path_set::Set{CartesianIndex{2}})
  can_visit::Vector{CartesianIndex{2}} = []
  for neighbour in NEIGHBOURS
    p = path[end] + neighbour
    if !checkbounds(Bool, data, p) || p ∈ path_set || data[p] == '#'
      continue
    end
    push!(can_visit, p)
  end
  if isempty(can_visit)
    return path[end][2] == size(data, 2) ? length(path) : 0
  end
  max_len = 0
  for v in can_visit
    nv = nothing
    if data[v] ∈ keys(SLOPES)
      nv = v + SLOPES[data[v]]
      if nv ∈ path_set
        continue
      end
    end
    push!(path, v)
    push!(path_set, v)
    if !isnothing(nv)
      push!(path, nv)
      push!(path_set, nv)
    end
    max_len = max(max_len, visit_part1(data, path, path_set))
    if !isnothing(nv)
      pop!(path)
      delete!(path_set, nv)
    end
    pop!(path)
    delete!(path_set, v)
  end
  max_len
end

function solve_part1(data, start::CartesianIndex{2})
  visit_part1(data, [start], Set{CartesianIndex{2}}([start]))
end

function visit_part2(data, path::Vector{CartesianIndex{2}}, path_set::Set{CartesianIndex{2}})
  can_visit::Vector{CartesianIndex{2}} = []
  for neighbour in NEIGHBOURS
    p = path[end] + neighbour
    if !checkbounds(Bool, data, p) || p ∈ path_set || data[p] == '#' || data[p] == '$'
      continue
    end
    push!(can_visit, p)
  end
  if isempty(can_visit)
    return path[end][2] == size(data, 2) ? length(path) : 0
  end
  max_len = 0
  for v in can_visit
    push!(path, v)
    push!(path_set, v)
    max_len = max(max_len, visit_part2(data, path, path_set))
    pop!(path)
    delete!(path_set, v)
  end
  max_len
end

function solve_part2(data, start::CartesianIndex{2})
  visit_part2(data, [start], Set{CartesianIndex{2}}([start]))
end

data = stack(Iterators.map(collect, eachline("$(@__DIR__)/example")))
start = CartesianIndex(findfirst(isequal('.'), data[:, 1]), 1)
println("Part 1: ", solve_part1(data, start) - 1)
println("Part 2: ", solve_part2(data, start) - 1)
