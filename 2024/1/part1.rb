#!/usr/bin/env ruby
puts ARGF.each_line.reduce(0) { |sum, line|
  line =~ /^[^\d]*(\d)/
  first_digit = $1
  line =~ /.*(\d)[^\d]*$/
  last_digit = $1
  sum + "#{first_digit}#{last_digit}".to_i
}
