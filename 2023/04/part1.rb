#!/usr/bin/env ruby

puts ARGF.each_line.reduce(0) { |sum, line|
  winning, ours = line.sub(/^Card +(\d+): +/, '').
    split(/ +\| +/).
    map { _1.split(/ +/).map(&:to_i) }
  num_winning = (winning & ours).size
  score = num_winning > 0 ? 2 ** (num_winning - 1) : 0
  sum + score
}
