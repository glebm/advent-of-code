#!/usr/bin/env ruby

WORDS = {
  'one' => 1,
  'two' => 2,
  'three' => 3,
  'four' => 4,
  'five' => 5,
  'six' => 6,
  'seven' => 7,
  'eight' => 8,
  'nine' => 9
}
WORDS_RE = "#{WORDS.keys.join('|')}"

def parse_digit(d)
  WORDS[d] || d.to_i
end

puts ARGF.each_line.reduce(0) { |sum, line|
  line =~ /^.*?(\d|#{WORDS_RE})/
  first_digit = $1
  line =~ /.*(\d|#{WORDS_RE}).*$/
  last_digit = $1
  sum + "#{parse_digit first_digit}#{parse_digit last_digit}".to_i
}
