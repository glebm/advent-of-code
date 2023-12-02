#!/usr/bin/env ruby

def nums(round)
  h = {}
  round.split(', ').each do |desc|
    desc =~ /(\d+) (red|green|blue)/
    h[$2] = $1.to_i
  end
  h
end

def game_power(game)
  max = {}
  game.split('; ').each do |round|
    h = nums(round)
    h.each do |k, v|
      max[k] = [max[k] || 0, v].max.to_i
    end
  end
  return 0 if max.size != 3
  max.values.reduce(1, :*)
end

puts ARGF.each_line.reduce(0) { |sum, line|
  line =~ /^Game (\d+): /
  game_num = $1.to_i
  sum + game_power(line[$~.end(0)...])
}
