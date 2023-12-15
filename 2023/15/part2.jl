#!/usr/bin/env julia

using LinearAlgebra
using StaticArrays # import Pkg; Pkg.add("StaticArrays")

focal_hash(str::AbstractString)::Int = foldl(str; init=0) do r, c
  (r + Int(c)) * 17 % 256
end

struct Lens
  label::String
  focal::Int
end

function step(boxes::SVector{256, Vector{Lens}}, step::AbstractString)
  label, val = match(r"^(\w+)[-=](-?\d+)?", step)
  index = focal_hash(label)
  box = boxes[index + 1]
  lens_pos = findfirst(box) do lens
    lens.label == label
  end
  if isnothing(val)
    !isnothing(lens_pos) && deleteat!(box, lens_pos)
  else
    val = parse(Int, val)
    if isnothing(lens_pos)
      push!(box, Lens(label, val))
    else
      box[lens_pos] = Lens(label, val)
    end
  end
  nothing
end

score(box::Vector{Lens})::Int =
  isempty(box) ? 0 : eachindex(box) ⋅ (lens.focal for lens in box)

steps = eachsplit(readline(), ",")
boxes = @SVector [Vector{Lens}() for _=1:256]
foreach(steps) do s
  step(boxes, s)
end
@time sum(eachindex(boxes) ⋅ Iterators.map(score, boxes)) |> println
