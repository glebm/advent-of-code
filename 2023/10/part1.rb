#!/usr/bin/env ruby

SYM_TO_DIR = {
  '|' => %i[north south], '-' => %i[west east],
  'L' => %i[north east], 'J' => %i[north west],
  '7' => %i[west south], 'F' => %i[east south], '.' => %i[],
}

class Point < Data.define(:x, :y)
  def west() = with(x: x - 1); def east() = with(x: x + 1)
  def north() = with(y: y - 1); def south() = with(y: y + 1);
  def to(dir) = send(dir)
end

class Data2D < Data.define(:array)
  def [](point) = array[point.y][point.x]
  def []=(point, value); array[point.y][point.x] = value end
  def w = array[0].size; def h = array.size

  def west?(point) = point.x > 0; def east?(point) = point.x + 1 < w
  def north?(point) = point.y > 0; def south?(point) = point.y + 1 < h
  def to?(dir, point) = send(:"#{dir}?", point)

  def each_position
    return to_enum(__method__) { w * h } unless block_given?
    (0...h).each { |y| (0...w).each { |x| yield Point.new(x, y) } }
  end
end

def adj(data, point)
  return to_enum(__method__, data, point) unless block_given?
  SYM_TO_DIR[data[point]].each { yield(point.to(_1)) if data.to?(_1, point) }
end

map = Data2D.new($<.readlines(chomp: true))
start = map.each_position.find { |point| map[point] == 'S' }

dist = Data2D.new(Array.new(map.h) { Array.new(map.w, -1) })
dist[start] = 0

queue = %i[north east south west].filter_map { |dir|
  next unless map.to?(dir, start)
  point = start.to(dir)
  [point, start] if adj(map, point).include?(start)
}
loop do
  point, prev = queue.shift
  if dist[point] != -1
    puts dist[point]
    exit 0
  end
  dist[point] = dist[prev] + 1
  adj(map, point).each { queue << [_1, point] if _1 != prev }
end
