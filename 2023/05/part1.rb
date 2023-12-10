#!/usr/bin/env ruby

MapEntry = Struct.new(:src_begin, :dst_begin, :len)

def read_map(input)
  result = []
  input.next
  loop do
    line = input.next.chomp
    raise StopIteration if line.empty?
    dst, src, len = line.scan(/\d+/).map(&:to_i)
    result << MapEntry.new(src, dst, len)
  end
  result.sort_by!(&:src_begin)
  result.reverse!
  result
end

def apply_map(src, entries)
  entry = entries.bsearch { |entry| src >= entry.src_begin }
  return src if entry.nil? || src >= entry.src_begin + entry.len
  entry.dst_begin + (src - entry.src_begin)
end

input = ARGF.each_line
seeds = input.next.scan(/\d+/).map(&:to_i)
input.next

7.times {
  map = read_map(input)
  seeds.map! { apply_map(_1, map) }
}

puts seeds.min
