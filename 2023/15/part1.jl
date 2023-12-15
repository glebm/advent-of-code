#!/usr/bin/env julia

function focal_hash(str::AbstractString)
  foldl(str; init=0) do r, c
    r += Int(c)
    r *= 17
    r % 256
  end
end

split(readline(), ",") .|> focal_hash |> sum |> println
