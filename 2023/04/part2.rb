#!/usr/bin/env ruby

card_wins = ARGF.each_line.map { |line|
  winning, ours = line.sub(/^Card +(\d+): +/, '').
    split(/ +\| +/).
    map { _1.split(/ +/).map(&:to_i) }
  (winning & ours).size
}

puts Array.new(card_wins.size).tap { |net|
  (0...net.size).reverse_each { |i|
    net[i] = net[i + 1..i + card_wins[i]].sum + 1
  }
}.sum
