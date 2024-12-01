#!/usr/bin/env ruby
lists = $<.map { _1.scan(/\d+/).map(&:to_i) }.transpose
lists.each(&:sort!)
puts lists[0].zip(lists[1]).lazy.map { |(a, b)| (a - b).abs }.sum
puts lists[0].lazy.map { |a|
  lo = lists[1].bsearch_index { |b| b >= a }
  next 0 if lo.nil?
  count = (lo...).lazy.take_while { _1 < lists[1].size && lists[1][_1] == a }.count
  a * count
}.sum
