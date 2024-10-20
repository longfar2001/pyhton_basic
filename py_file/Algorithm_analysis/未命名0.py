#%%
from typing import List

routes = [[1, 2, 7], [3, 6, 7]]
S = 1
T = 6

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if not routes:
            return -1

        if S == T:
            return 0

        routes = list(map(set, routes))
        rows = len(routes)

        start = 0
        end = 0
        # graph是一个二维数组，graph[v][u]表示节点v到u的代价
        graph = [[float('inf') for _ in range(rows)] for _ in range(rows)]
        for i, row_route in enumerate(routes):
            if S in row_route and T in row_route:
                return 1

            if S in row_route:
                start = i
            if T in row_route:
                end = i

            # 自己到自己的代价为0
            graph[i][i] = 0
            for j in range(i + 1, rows):
                if any([route in routes[j] for route in row_route]):
                    # 自己到其他点的代价为1
                    graph[i][j] = 1
                    graph[j][i] = 1

        # 典型的floyd三层循环
        for k in range(rows):
            for i in range(rows):
                for j in range(rows):
                    graph[i][j] = min(graph[i][j], graph[i][k] + graph[k][j])

        # 注意这里需要加1，是因为上面我们定义的起始点到其他的代价为1，因此还需要加上自己，才表示总的需要换乘的线路
        return -1 if graph[start][end] == float('inf') else graph[start][end] + 1


Solution().numBusesToDestination(routes = [[1, 2, 7], [3, 6, 7]],S = 1,T = 6)

#%%
from typing import List


class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int,T: int) -> int:
        if not routes:
            return -1

        if S == T:
            return 0

        rows = len(routes)

        start = 0
        end = 0
        # graph是一个二维数组，graph[v][u]表示节点v到u的代价
        graph = [[float('inf') for _ in range(rows)] for _ in range(rows)]
        for i, row_route in enumerate(routes):
            if S in row_route and T in row_route:
                return 1

            if S in row_route:
                start = i
            if T in row_route:
                end = i

            # 自己到自己的代价为0
            graph[i][i] = 0
            for j in range(i + 1, rows):
                if any([route in routes[j] for route in row_route]):
                    # 自己到其他点的代价为1
                    graph[i][j] = 1
                    graph[j][i] = 1

        # 典型的dijkstra算法步骤
        not_visited_nodes = [i for i in range(rows) if i != start]
        for _ in range(rows - 1):
            min_cost = float('inf')
            min_node = -1
            for not_visited_node in not_visited_nodes:
                if graph[start][not_visited_node] <= min_cost:
                    min_cost = graph[start][not_visited_node]
                    min_node = not_visited_node

            not_visited_nodes.remove(min_node)
            for not_visited_node in not_visited_nodes:
                graph[start][not_visited_node] = min(graph[start][not_visited_node],
                    graph[start][min_node] + graph[min_node][not_visited_node])

        # 注意这里需要加1，是因为上面我们定义的起始点到其他的代价为1，因此还需要加上自己，才表示总的需要换乘的线路
        return -1 if graph[start][end] == float('inf') else graph[start][end] + 1

Solution().numBusesToDestination(routes = [[1, 2, 7], [3, 6, 7]],S = 1,T = 6)

Solution().numBusesToDestination(routes = [[1,2,3,4,5,6,7],[3,8,4],[8,9,10,11,12],[3,2,13,14,9,8],[12,15,16,17],[4,5,19,18,17,12,8]],S = 1,T = 12)

#%%
from collections import deque
from collections import defaultdict
# BFS
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S:int, T:int) -> int:
        # 每个车站可以乘坐的公交车
        stations = defaultdict(set)
        for i, stops in enumerate(routes):
            for stop in stops:
                stations[stop].add(i)
        # 每个公交车线路可以到达的车站
        routes = [set(x) for x in routes]

        q = deque([(S, 0)])
        # 已经乘坐了的公交车
        buses = set()
        # 已经到达了的车站
        stops = {S}
        while q:
            pos, cost = q.popleft()
            if pos == T:
                return cost
            # 当前车站中尚未乘坐的公交车
            for bus in stations[pos] - buses:
                # 该公交车尚未到达过的车站
                for stop in routes[bus] - stops:
                    buses.add(bus)
                    stops.add(stop)
                    q.append((stop, cost + 1))
        return -1

Solution().numBusesToDestination(routes = [[1,2,3,4,5,6,7,8],[3,9,4],[9,10,11,12,13],[13,14,15,16]],
                                 S = 1,T = 9)

#%%
def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
    if not routes:
        return -1
    if S == T:
        return 0
    routes = list(map(set, routes))
    rows = len(routes)
    start = 0
    end = 0
    graph = [[float('inf') for _ in range(rows)] for _ in range(rows)]
    for i, row_route in enumerate(routes):
        if S in row_route and T in row_route:
            return 1
        if S in row_route:
            start = i
        if T in row_route:
            end = i
        graph[i][i] = 0
        for j in range(i + 1, rows):
            if any([route in routes[j] for route in row_route]):
                graph[i][j] = 1
                graph[j][i] = 1
    for k in range(rows):
        for i in range(rows):
            for j in range(rows):
                graph[i][j] = min(graph[i][j], graph[i][k] + graph[k][j])
    return -1 if graph[start][end] == float('inf') else graph[start][end] + 1

#%%
def numBusesToDestination(self, routes: List[List[int]], S: int,T: int) -> int:
    if not routes:
        return -1
    if S == T:
        return 0
    rows = len(routes)
    start = 0
    end = 0
    graph = [[float('inf') for _ in range(rows)] for _ in range(rows)]
    for i, row_route in enumerate(routes):
        if S in row_route and T in row_route:
            return 1
        if S in row_route:
            start = i
        if T in row_route:
            end = i
        graph[i][i] = 0
        for j in range(i + 1, rows):
            if any([route in routes[j] for route in row_route]):
                graph[i][j] = 1
                graph[j][i] = 1
    not_visited_nodes = [i for i in range(rows) if i != start]
    for _ in range(rows - 1):
        min_cost = float('inf')
        min_node = -1
        for not_visited_node in not_visited_nodes:
            if graph[start][not_visited_node] <= min_cost:
                min_cost = graph[start][not_visited_node]
                min_node = not_visited_node
        not_visited_nodes.remove(min_node)
        for not_visited_node in not_visited_nodes:
            graph[start][not_visited_node] = min(graph[start][not_visited_node],
                graph[start][min_node] + graph[min_node][not_visited_node])
    return -1 if graph[start][end] == float('inf') else graph[start][end] + 1
