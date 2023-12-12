#!/usr/bin/env ruby

def set_last_part!(chunks, n)
  chunks[-1] = chunks[-1].dup.tap { _1[-1] = [_1[-1][0], n]; _1.pop if _1[-1][1].zero? }
  chunks.pop if chunks[-1].empty?
end

def decrement_last_part!(chunks, n = 1) = set_last_part!(chunks, chunks[-1][-1][1] - n)

def pop_chunk_part(chunks, n = 1)
  chunks = chunks.dup
  chunks[-1] = chunks[-1].dup.tap { _1.pop(n) }
  chunks.pop if chunks[-1].empty?
  chunks
end

def decrement_last_seq(seq, n)
  seq.dup.tap { _1[-1] -= n }
end

def pop(arr, n = 1) = arr[0, arr.size - n]

# .###.
def case_single_hash(chunks, seq, depth, hash_len)
  return 0 unless seq[-1] == hash_len
  solve(pop(chunks), pop(seq), depth + 1)
end

# .???.
def case_single_q(chunks, seq, depth, q_len)
  if_all_dot = solve(pop(chunks), seq, depth + 1)
  if_some_hash =
    if seq[-1] > q_len
      0
    else
      # Set prefix to '#':
      hash_prefix = solve(pop(chunks), pop(seq), depth + 1)
      # Any '#' infix of length `seq[-1]` preceded by '.', continue with the remaining '?' prefix:
      hash_infix = (0..(q_len - seq[-1] - 1)).lazy.map { |rem|
        solve(chunks.dup.tap { set_last_part!(_1, rem) }, pop(seq), depth + 1)
      }.sum
      hash_prefix + hash_infix
    end
  if_all_dot + if_some_hash
end

# *?#.
def case_last_hash(chunks, seq, depth, hash_len)
  if hash_len > seq[-1]
    0
  elsif hash_len == seq[-1]
    solve(pop_chunk_part(chunks).tap { decrement_last_part!(_1) }, pop(seq), depth + 1)
  else # hash_len < seq[-1]
    q_len = chunks[-1][-2][1]
    if q_len + hash_len == seq[-1]
      if chunks[-1].size == 2
        solve(pop(chunks), pop(seq), depth + 1)
      else
        0
      end
    elsif q_len + hash_len < seq[-1]
      if chunks[-1].size == 2
        0
      else
        solve(pop_chunk_part(chunks, 2), decrement_last_seq(seq, q_len + hash_len), depth + 1)
      end
    else # q_len + hash_len > seq[-1]
      solve(pop_chunk_part(chunks).tap { decrement_last_part!(_1, seq[-1] - hash_len + 1) }, pop(seq), depth + 1)
    end
  end
end

# *#?.
def case_last_q(chunks, seq, depth, q_len)
  result = 0
  chunks_without_q = pop_chunk_part(chunks)
  # We can always choose a '#' prefix <= seq[-1] and continue
  result += (0..[q_len, seq[-1]].min).lazy.map { |hash_prefix_len|
    solve(chunks_without_q, decrement_last_seq(seq, hash_prefix_len), depth + 1)
  }.sum

  if q_len > seq[-1]
    # Any '#' infix of length `seq[-1]` preceded by '.', continue with the remaining '?' prefix:
    result += (0..(q_len - seq[-1] - 1)).lazy.map { |rem|
      solve(chunks.dup.tap { set_last_part!(_1, rem) }, pop(seq), depth + 1)
    }.sum
  end
  result
end

def do_solve(chunks, seq, depth)
  if chunks.empty?
    return 1 if seq.empty?
    return 0
  end
  if seq.empty?
    return 1 if chunks.all? { _1 in [['?', _]] }
    return 0
  end
  case chunks[-1]
  in [['#', hash_len]] then case_single_hash(chunks, seq, depth, hash_len)
  in [['?', q_len]] then case_single_q(chunks, seq, depth, q_len)
  in [*, ['#', hash_len]] then case_last_hash(chunks, seq, depth, hash_len)
  in [*, ['?', q_len]] then case_last_q(chunks, seq, depth, q_len)
  end
end

CACHE = {}
def solve(chunks, seq, depth = 0)
  CACHE[[chunks, seq]] ||= begin
    log = false
    #log = true
    puts "#{'  ' * depth}solve #{chunks.inspect} #{seq.inspect}" if log
    do_solve(chunks, seq, depth).tap { puts "#{'  ' * depth}=> #{_1}" if log }
  end
end

n = 5
puts $<.map { |line|
  pattern, seq = line.chomp.split(' ')
  pattern = ([pattern] * n).join('?').chars
  seq = seq.split(',').map(&:to_i)
  seq *= n

  # A list of [char, len]
  chunks = pattern.chunk { _1 }.map { [_1, _2.count] }

  # Sequences of /[#?]+/
  chunks = chunks.chunk { _1[0] != '.' }.filter { _1[0] }.map { _1[1] }
  solve(chunks, seq) #.tap { puts "#{_1} #{line}"}
}.sum

