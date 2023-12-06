#!/usr/bin/env ruby

# t = total time, d = max distance
t, d = ARGF.each_line.map { _1.gsub(/[^\d]+/, '').to_i }

# If we hold for x seconds (x ∈ ℕ: x < t), then:
# dist(x) = (t - x) * x = -x² + tx
# dist(x) > d <=> -x² + tx - d > 0
#
# `dist` is an inverted parabola, it is positive between
# its roots r1 and r2 (r1 < r2).
#
# Its roots are:
# 1. Positive because t is positive.
# 2. Less than t because otherwise -x² + tx < 0.
#
# When the roots are integers (e.g. t = 8, d = 12),
# we cannot count them as part of the solution (strict inequality).
# We work around that by subtracting/adding a small number.
#
# The number of positive integer points < t,
# excluding integral root, is:
# floor(r2 - eps) - ceil(r1 + eps) + 1

# Check if the roots exist, i.e. b² > 4ac
unless t ** 2 > 4 * d
  puts 0
  return
end

# Roots of ax² + bx + c are (-b ± sqrt(b² - 4ac)) / 2a
r1 = (t - Math.sqrt(t ** 2 - 4 * d)) / 2
r2 = (t + Math.sqrt(t ** 2 - 4 * d)) / 2
eps = 0.00000001
puts (r2 - eps).floor - (r1 + eps).ceil + 1
