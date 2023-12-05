#!/usr/bin/env ruby

MapEntry = Struct.new(:src_begin, :dst_begin, :len) do
  def src_end = src_begin + len
end

def read_map(input)
  result = []
  input.next
  loop do
    line = input.next.chomp
    raise StopIteration if line.empty?
    dst, src, len = line.scan(/\d+/).map(&:to_i)
    result << MapEntry.new(src, dst, len)
  end
  result
end

def map_entry(entry, src) = entry.dst_begin + (src - entry.src_begin)

def apply_map(src_range, entries)
  mapped = []
  matched = false
  entries.each do |entry|
    next if entry.src_begin >= src_range.end || src_range.begin >= entry.src_end
    matched = true
    mapped << (map_entry(entry, [src_range.begin, entry.src_begin].max)...
      map_entry(entry, [src_range.end, entry.src_end].min))
    mapped.concat(apply_map((src_range.begin...entry.src_begin), entries)) if src_range.begin < entry.src_begin
    mapped.concat(apply_map((entry.src_end...src_range.end), entries)) if entry.src_end < src_range.end
  end
  mapped << src_range unless matched
  mapped
end

input = ARGF.each_line
seeds = input.next.scan(/\d+/).map(&:to_i).each_slice(2).map { (_1..._1 + _2) }
input.next

7.times {
  map = read_map(input)
  seeds = seeds.flat_map { apply_map(_1, map) }.uniq
}

puts seeds.min_by(&:first).first
