#!/usr/bin/env julia

function focal_hash(str::AbstractString)
  foldl(str; init=0) do r, c
    r += Int(c)
    r *= 17
    r % 256
  end
end

struct Lens
  label::String
  focal::Int
end

const Box = Vector{Lens}

function step(boxes::Vector{Box}, step::AbstractString)
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
  boxes
end

function score((boxnum, box)::Tuple{Int, Box})
  box |> enumerate .|> ((slot, lens)::Tuple{Int, Lens} ->
    boxnum * slot * lens.focal) |> sum
end

steps = split(readline(), ",")
boxes = [Box() for _=1:256]

foldl(step, steps, init=boxes) |> enumerate .|> score |> sum |> println
