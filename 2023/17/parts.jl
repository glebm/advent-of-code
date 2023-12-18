#!/usr/bin/env julia
using DataStructures

const MAX_COST = typemax(Int)

@views function dijkstra(data::Matrix{Int}, range::UnitRange{Int})
  dist = fill(MAX_COST, 4, size(data)...)
  prev = fill(CartesianIndex(-1, -1, -1), size(dist))
  queue = PriorityQueue{CartesianIndex{3},Int}()
  for i in CartesianIndices(axes(dist))
    queue[i] = MAX_COST
  end
  for source in (CartesianIndex(2, 1, 1), CartesianIndex(3, 1, 1))
    dist[source] = queue[source] = 0
  end
  Σ = (cumsum(data, dims=1), cumsum(transpose(data), dims=1))
  while !isempty(queue)
    u = dequeue!(queue)
    xy_u = CartesianIndex(u[2], u[3])
    xy_u == size(data) && return
    dist_u = dist[u]
    dist_u == MAX_COST && continue
    for dir in (mod1(u[1] + 1, 4), mod1(u[1] + 3, 4))
      dim = iseven(dir) ? 1 : 2
      t_dir = dir == 2 || dir == 3 ? 1 : -1
      t_begin = u[dim+1] + t_dir * first(range)
      t_end = clamp(u[dim+1] + t_dir * last(range), 1, size(data, dim))
      for t in (t_begin:t_dir:t_end)
        xy = dim == 1 ? CartesianIndex(t, xy_u[2]) : CartesianIndex(xy_u[1], t)
        v = CartesianIndex(dir, xy)
        dist_v = get(queue, v, -1)
        dist_v == -1 && continue
        Σ_min, Σ_max = minmax(t, xy_u[dim])
        alt = dist_u +
              Σ[dim][Σ_max, xy[3-dim]] - Σ[dim][Σ_min, xy[3-dim]] +
              (t_dir == -1 ? data[xy] - data[xy_u] : 0)
        if alt < dist_v
          queue[v] = dist[v] = alt
          prev[v] = u
        end
      end
    end
  end
  dist, prev
end

@views function print_path(
  data::Matrix{Int}, prev::Array{CartesianIndex{3},3}, dest::CartesianIndex{3})
  grid = data .|> (x -> '0' + x)
  cur = dest
  while true
    p = prev[cur]
    p[1] == -1 && break
    if p[2] == cur[2]
      if p[3] < cur[3]
        grid[p[2], p[3]:cur[3]] .= 'v'
      else
        grid[p[2], cur[3]:p[3]] .= '^'
      end
    else
      if p[2] < cur[2]
        grid[p[2]:cur[2], p[3]] .= '>'
      else
        grid[cur[2]:p[2], p[3]] .= '<'
      end
    end
    cur = p
  end
  grid[1, 1] = '0' + data[1, 1]
  foreach(col -> println(prod(col)), eachcol(grid))
end

@views data = stack([c - '0' for c in line] for line in eachline())
for (i, range) in enumerate((1:3, 4:10))
  dist, prev = dijkstra(data, range)
  dest = CartesianIndex(argmin(dist[:, size(data)...]), size(data)...)
  # print_path(data, prev, dest)
  println("Part ", i, " ", dist[dest])
end
