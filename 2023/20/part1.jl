#!/usr/bin/env julia

using DataStructures

struct Mod
  type::Char
  dest::Vector{Integer}
end
struct Mods
  all::Vector{Mod}
  conj::Vector{Int}
  conj_sources::Vector{Vector{Int}}
end
const Signal = Tuple{Bool,Int,Int}
struct State
  flip::Vector{Bool}
  conj::Vector{Vector{Bool}}
  sent::Array{Int}
  queue::Queue{Signal}
end

@inbounds @views function send_hi(mods::Mods, from::Int, to::Int, state::State)
  state.sent[2] += 1
  # println("$from -high-> $to")
  to ∉ keys(mods.all) && return
  mod::Mod = mods.all[to]
  if mod.type == '&'
    conj_i = mods.conj[to]
    conj = state.conj[conj_i]
    conj[mods.conj_sources[conj_i][from]] = true
    pulse = !all(conj)
    foreach(x -> enqueue!(state.queue, (pulse, to, x)), mod.dest)
  end
end

@inbounds @views function send_lo(mods::Mods, from::Int, to::Int, state::State)
  state.sent[1] += 1
  # println("$from -low-> $to")
  to ∉ keys(mods.all) && return
  mod::Mod = mods.all[to]
  if mod.type == '%'
    pulse = state.flip[to] = !state.flip[to]
    foreach(x -> enqueue!(state.queue, (pulse, to, x)), mod.dest)
  else
    conj_i = mods.conj[to]
    conj = state.conj[conj_i]
    conj[mods.conj_sources[conj_i][from]] = false
    foreach(x -> enqueue!(state.queue, (true, to, x)), mod.dest)
  end
end

init_state() = State(
  falses(length(mods.all)),
  [falses(count(x -> x != 0, src)) for src in mods.conj_sources],
  [0, 0], Queue{Signal}())

@inbounds @views function solve_part1(mods::Mods, broadcaster::Array{Int})
  state = init_state()
  for _ in 1:1000
    state.sent[1] += 1
    foreach(x -> enqueue!(state.queue, (false, -1, x)), broadcaster)
    while !isempty(state.queue)
      (hi, from, to) = dequeue!(state.queue)
      if hi
        send_hi(mods, from, to, state)
      else
        send_lo(mods, from, to, state)
      end
    end
    # println()
  end
  prod(state.sent)
end

@inbounds @views function solve_part2(mods::Mods, broadcaster::Array{Int}, rx::Int)
  state = init_state()
  for i in Iterators.countfrom(1)
    foreach(x -> enqueue!(state.queue, (false, -1, x)), broadcaster)
    while !isempty(state.queue)
      (hi, from, to) = dequeue!(state.queue)
      if hi
        send_hi(mods, from, to, state)
      else
        to == rx && return i
        send_lo(mods, from, to, state)
      end
    end
    (i % 1000000) == 0 && println("Pressed the button ", i ÷ 1000000, " million times")
  end
end

# Convert the map to arrays and indices for efficiency
function convert_input(sym_mods)
  syms::Array{Symbol} = [k for k in keys(sym_mods) if k != :broadcaster]
  sym_to_idx::Dict{Symbol,Int} =
    Dict(k => i for (i, k) in enumerate(syms) if k != :broadcaster)
  all_mods = let
    sinks = 0
    for (k, (_, dests)) in sym_mods, dest in dests
      dest ∈ keys(sym_to_idx) && continue
      sinks += 1
      sym_to_idx[dest] = length(syms) + sinks
    end
    [
      let (type, dest) = sym_mods[sym]
        Mod(type, map(x -> sym_to_idx[x], dest))
      end for sym in syms
    ]
  end
  num_conj = 0
  conj = [(m.type == '&' ? (num_conj += 1) : 0)
          for (_, m) in enumerate(all_mods)]
  conj_sources = [zeros(length(all_mods)) for _ in 1:num_conj]
  for (dest_i, conj_i) in enumerate(conj)
    conj_i == 0 && continue
    num_src = 0
    for (src_i, m) in enumerate(all_mods)
      dest_i ∉ m.dest && continue
      conj_sources[conj_i][src_i] = (num_src += 1)
    end
  end
  Mods(all_mods, conj, conj_sources), sym_to_idx
end

sym_mods = Dict(let (k, v) = split(line, " -> ")
  Symbol(k[1] == 'b' ? k : k[2:end]) => (
    k[1], map(Symbol, eachsplit(v, ", ")))
end for line in eachline())
# end for line in eachline("$(@__DIR__)/example"))
mods, sym_to_idx = convert_input(sym_mods)

broadcaster = [sym_to_idx[d] for d in sym_mods[:broadcaster][2]]
println("Part 1: ", solve_part1(mods, broadcaster))
if :rx ∈ keys(sym_to_idx)
  println("Part 2: ", solve_part2(mods, broadcaster, sym_to_idx[:rx]))
end
