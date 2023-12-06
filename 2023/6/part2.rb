#!/usr/bin/env ruby

total_time, max_distance = ARGF.each_line.map { _1.gsub(/[^\d]+/, '').to_i }

puts (0...total_time).count { |time_to_hold|
  speed = time_to_hold
  remaining_time = total_time - time_to_hold
  dist = speed * remaining_time
  dist > max_distance
}
