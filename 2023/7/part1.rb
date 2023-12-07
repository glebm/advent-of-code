#!/usr/bin/env ruby

hands = $<.each_line.map { h, s = _1.split(' '); [h.chars.map(&:to_sym), s.to_i] }

COMBOS = %i[five four full_house three two_pair one_pair high_card none]
CARDS = %i[A K Q J T 9 8 7 6 5 4 3 2]

def combo(hand)
  case hand.group_by { _1 }.transform_values!(&:count).values.sort
  in [5] then :five
  in [1, 4] then :four
  in [2, 3] then :full_house
  in [1, 1, 3] then :three
  in [1, 2, 2] then :two_pair
  in [1, 1, 1, 2] then :one_pair
  in [1, 1, 1, 1, 1] then :high_card
  else :none
  end
end

puts hands.sort_by { |(hand, score)|
  [COMBOS.index(combo(hand)), *(0...5).map { CARDS.index(hand[_1]) }]
}.each_with_index.map { |(hand, score), i| score * (hands.size - i) }.sum
