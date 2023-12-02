#!/usr/bin/env ruby

LIMITS = {
  'red' => 12,
  'green' => 13,
  'blue' => 14
}

def valid_round(round)
  round.split(', ').all? do |desc|
    desc =~ /(\d+) (red|green|blue)/
    $1.to_i <= LIMITS[$2]
  end
end

def valid_game(game)
  game.split('; ').all? { |round| valid_round(round) }
end

puts ARGF.each_line.reduce(0) { |sum, line|
  line =~ /^Game (\d+): /
  game_num = $1.to_i
  valid_game(line[$~.end(0)...]) ? sum + game_num : sum
}
