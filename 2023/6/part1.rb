#!/usr/bin/env ruby

times, distances = ARGF.each_line.map { _1.scan(/\d+/).map(&:to_i) }

puts times.zip(distances).reduce(1) { |product, (total_time, max_distance)|
  product * (0...total_time).count { |time_to_hold|
    speed = time_to_hold
    remaining_time = total_time - time_to_hold
    dist = speed * remaining_time
    dist > max_distance
  }
}
