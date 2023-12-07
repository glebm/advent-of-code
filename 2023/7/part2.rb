#!/usr/bin/env ruby

hands = $<.each_line.map { h, s = _1.split(' '); [h.chars.map(&:to_sym), s.to_i] }

COMBOS = %i[five four full_house three two_pair one_pair high_card none]
CARDS = %i[A K Q T 9 8 7 6 5 4 3 2 J]

def combo(hand)
  j = hand.count(:J)
  c = (hand - [:J]).group_by { _1 }.transform_values!(&:count).values.sort
  max = c.empty? ? 0 : c[-1]
  return :five if max + j == 5
  return :four if max + j == 4
  return :full_house if !c.empty? && c[0] == 2 && max + j == 3
  return :three if max + j == 3
  return :two_pair if c.size >= 2 && j == 0 && c[-2] == 2 && max == 2
  return :one_pair if max + j == 2
  return :high_card if c.size == 5
  return :none
end

puts hands.sort_by! { |(hand, score)|
  [COMBOS.index(combo(hand)), *(0...5).map { CARDS.index(hand[_1]) }]
}.each_with_index.map { |(hand, score), i| score * (hands.size - i) }.sum
