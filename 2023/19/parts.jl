#!/usr/bin/env julia

struct UnconditionalRule
  target::String
end
struct ConditionalRule
  category::Int
  op::Symbol
  rarg::Int
  target::String
end
const Rule = Union{UnconditionalRule,ConditionalRule}

struct Workflow
  name::String
  rules::Vector{Rule}
end

const Part = NTuple{4,Int}
const CHAR_IDX = Dict('x' => 1, 'm' => 2, 'a' => 3, 's' => 4)

function parse_rule(str)
  m = match(r"(\w)([><])([-\d]+):(\w+)", str)
  if isnothing(m)
    UnconditionalRule(str)
  else
    ConditionalRule(CHAR_IDX[m[1][1]], Symbol(m[2]), parse(Int, m[3]), m[4])
  end
end

function parse_workflow(line)
  name, contents = match(r"(\w+)\{(.*?)\}", line)
  Workflow(name, map(parse_rule, split(contents, ',')))
end

function parse_workflows(lines)
  workflows = map(parse_workflow, lines)
  by_name = Dict(workflow.name => i for (i, workflow) in enumerate(workflows))
  workflows, by_name
end

# Part 1
rule_matches(part::Part, rule::UnconditionalRule) = true
rule_matches(part::Part, rule::ConditionalRule) =
  getfield(Base, rule.op)(part[rule.category], rule.rarg)
function accepted(wfs::Vector{Workflow}, wf_by_name::Dict{String,Int}, part::Part)
  (wf_i, rule_i) = (wf_by_name["in"], 1)
  while true
    wf = wfs[wf_i]
    rule = wf.rules[rule_i]
    if rule_matches(part, rule)
      rule.target == "A" && return true
      rule.target == "R" && return false
      (wf_i, rule_i) = (wf_by_name[rule.target], 1)
    else
      rule_i += 1
    end
  end
end

function solve_part1(workflows::Vector{Workflow},
  wf_by_name::Dict{String,Int}, parts_src)
  parts = [
    Part(parse(Int, m.match) for m in eachmatch(r"\d+", line))
    for line in parts_src
  ]
  sum(
    sum(part) for part in parts
    if accepted(workflows, wf_by_name, part)
  )
end

# Part 2
const IntervalSet = Vector{UnitRange{Int}}
const IntervalSets = NTuple{4,IntervalSet}

function remove_subset(intervals::IntervalSet, exclude::UnitRange{Int})
  result = []
  push_nonempty!(i) = !isempty(i) && push!(result, i)
  for r in intervals
    push_nonempty!(first(r):first(exclude)-1)
    push_nonempty!(last(exclude)+1:last(r))
  end
  result
end

function shrink_ranges(rule::Rule, ranges::IntervalSets, if_true::Bool)
  op = rule.op
  adj = 0
  if !if_true
    op = (op == :> ? :< : :>)
    adj = 1
  end
  exclude = op == :> ? (1:rule.rarg-adj) : (rule.rarg+adj:4000)
  IntervalSets(
    i == rule.category ? remove_subset(ranges[rule.category], exclude) : r
    for (i, r) in enumerate(ranges))
end

ranges_value(ranges::IntervalSets) = prod(r -> sum(length, r), ranges)
ranges_valid(ranges::IntervalSets) = !any(r -> any(isempty, r), ranges)

const RuleId = NTuple{2,Int}
function visit(workflows::Vector{Workflow}, wf_by_name::Dict{String,Int},
  cur::RuleId, ranges::IntervalSets, depth::Int=0, log_debug=false)
  wf = workflows[cur[1]]
  rule = wf.rules[cur[2]]
  if isa(rule, UnconditionalRule)
    rule.target == "R" && return 0
    if rule.target == "A"
      log_debug && log_pre("⭐", wf, rule, ranges, depth)
      result = ranges_value(ranges)
      log_debug && log_value(result, depth)
      return result
    end
    return visit(workflows, wf_by_name,
      (wf_by_name[rule.target], 1), ranges, depth + 1)
  end

  result_if_true = 0
  if rule.target != "R"
    ranges_if_true = shrink_ranges(rule, ranges, true)
    if ranges_valid(ranges_if_true)
      log_debug && log_pre("✔", wf, rule, ranges_if_true, depth)
      if rule.target == "A"
        result_if_true = ranges_value(ranges_if_true)
      else
        result_if_true = visit(workflows, wf_by_name,
          (wf_by_name[rule.target], 1), ranges_if_true, depth + 1)
      end
    end
  end

  result_if_false = 0
  ranges_if_false = shrink_ranges(rule, ranges, false)
  if ranges_valid(ranges_if_false)
    log_debug && log_pre("✖", wf, rule, ranges_if_false, depth)
    result_if_false = visit(workflows, wf_by_name,
      (cur[1], cur[2] + 1), ranges_if_false, depth + 1)
  end

  log_debug && log_value(result_if_false + result_if_true, depth)
  result_if_false + result_if_true
end

function solve_part2(workflows::Vector{Workflow}, wf_by_name::Dict{String,Int})
  ranges = ([1:4000], [1:4000], [1:4000], [1:4000])
  src = (wf_by_name["in"], 1)
  visit(workflows, wf_by_name, src, ranges)
end

# Debug logging
interval_set_str(is::IntervalSet) = join(("$(first(i)):$(last(i))" for i in is), "∪")
ranges_str(ranges::IntervalSets) = "($(join((interval_set_str(r) for r in ranges), "; ")))"
rule_str(r::UnconditionalRule) = "$(r.target)"
rule_str(r::ConditionalRule) = "$(r.category)$(r.op)$(r.rarg):$(r.target)"
log_pre(type, wf, rule, ranges, depth) = println(
  " │"^depth, " ┞", type, " ", wf.name, " ", rule_str(rule), " ", ranges_str(ranges))
log_value(value, depth) = println(" │"^depth, " ╰", "▶ ", value)

lines = readlines()
blank = findfirst(isempty, lines)
@views workflows_src, parts_src = lines[1:blank-1], lines[blank+1:end]
workflows, wf_by_name = parse_workflows(workflows_src)

println("Part 1: ", solve_part1(workflows, wf_by_name, parts_src))
println("Part 2: ", solve_part2(workflows, wf_by_name))
