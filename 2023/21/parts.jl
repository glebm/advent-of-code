#!/usr/bin/env julia

NEIGHBOURS = (CartesianIndex(0, -1), CartesianIndex(1, 0),
  CartesianIndex(0, 1), CartesianIndex(-1, 0))

function solve(data, start, num_steps)
  N = size(data, 1)
  δ = [0, zeros(Int, N), zeros(Int, N)]
  prev = Set{CartesianIndex{2}}()
  cur = Set{CartesianIndex{2}}()
  next = Set{CartesianIndex{2}}()
  push!(next, start)
  ok_step_rem2 = num_steps % 2
  result::Int = ok_step_rem2 == 0
  loop_limit = 10N - ok_step_rem2
  j = 0
  for i in 1:loop_limit-1
    prev, cur, next = cur, next, prev
    empty!(next)
    for s in cur, d in NEIGHBOURS
      pos = s + d
      if data[mod1(pos[1], size(data, 1)), mod1(pos[2], size(data, 2))] == '.' &&
         pos ∉ prev && pos ∉ cur
        push!(next, pos)
      end
    end
    if i % 2 == ok_step_rem2
      j = mod1(j + 1, N)
      δ1_prev = δ[1]
      δ2_prev = δ[2][j]
      δ[1] = length(next)
      δ[2][j] = δ[1] - δ1_prev
      δ[3][j] = δ[2][j] - δ2_prev
      result += δ[1]
    end
    num_steps == i && return result
  end
  for _ in loop_limit:2:num_steps
    j = mod1(j + 1, N)
    δ2 = δ[2][j] + δ[3][j]
    δ1 = δ[1] + δ2
    δ[1] = δ1
    δ[2][j] = δ2
    result += δ1
  end
  result
end

data = stack(Iterators.map(collect, eachline()))
start = findfirst(i -> data[i] == 'S', CartesianIndices(data))
data[start] = '.'
for (i, num_steps) in enumerate((64, 26501365))
  println("Part $i (num_steps=$num_steps): ", solve(data, start, num_steps))
end
