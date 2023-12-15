#!/usr/bin/env julia

focal_hash(str::AbstractString)::Int = foldl(str; init=0) do r, c
  (r + Int(c)) * 17 % 256
end

eachsplit(readline(), ",") .|> focal_hash |> sum |> println
