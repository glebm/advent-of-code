#!/usr/bin/env ruby

require 'z3'

stones = $<.readlines(chomp: true).map { _1.scan(/\-?\d+/).map(&:to_i) }

solver = Z3::Solver.new
px = Z3.Real("px")
py = Z3.Real("py")
pz = Z3.Real("pz")
vx = Z3.Real("vx")
vy = Z3.Real("vy")
vz = Z3.Real("vz")
stones.each_with_index do |(spx, spy, spz, svx, svy, svz), i|
  t = Z3.Real("t#{i}")
  solver.assert px + vx * t == spx + svx * t
  solver.assert py + vy * t == spy + svy * t
  solver.assert pz + vz * t == spz + svz * t
  solver.assert t > 0
end

if solver.satisfiable?
  puts solver.model
  puts (solver.model[px] + solver.model[py] + solver.model[pz]).simplify
else
  puts "no solution"
end
