#Thu vien
from collections import deque
import pandas as pd
import numpy as np
import heapq
import itertools
import copy
import Database

##################################################################################################################################
# Cây khung nhỏ nhất    : Boruvka(Sollin), Kruskal, Prim
# Đường đi ngắn nhất    : Dijkstra, Bellman-Ford
# Chu trình euler       : Fleury, Hierholzer
# Chu trình Hamintont   : Bellman-Held-Karp(dynamic programing), Branch and Bound
# Tô màu		        : Greedy Coloring 
# Đỉnh cắt 	            : Critical Vertices 
# Cạnh cắt	            : Bridges 
##################################################################################################################################
def dfs(adj_list, vertices, start=None):
    seq = []
    if start is None:
        start = vertices[0]
    # Khởi tạo 
    stack = [start]
    visited = [start]

    while stack:
        vertex = stack.pop()
        seq.append(vertex)

        for neighbor, _ in adj_list.get(vertex, []):
            if neighbor not in visited and neighbor in vertices:
                stack.append(neighbor)
                visited.append(neighbor)
    
    return seq
    
def all_components_dfs(adj_list, vertices):
    components = []
    _vertices = vertices.copy()

    while _vertices:
        component = dfs(adj_list,_vertices)
        components.append(component)

        for v in component:
            _vertices.remove(v)
    
    return components

def find_path_dfs(adj_list, vertices, start, end):
    if start not in adj_list or end not in adj_list:
        return None
    stack = [start]
    visited = [start]
    parent = {start: None}

    while stack:
        vertex = stack.pop()

        if vertex == end:
            path = []
            while vertex is not None:
                path.append(vertex)
                vertex = parent[vertex]
            path.reverse()
            return path  
        
        for neighbor, _ in adj_list.get(vertex, []):
            if neighbor not in visited:
                parent[neighbor] = vertex  # Ghi lại cha của neighbor
                stack.append(neighbor)
                visited.append(neighbor)
    
    return None

##################################################################################################################################
def bfs(adj_list, vertices, start=None):
    seq = []
    if start is None:
        start = vertices[0]
    
    # Khởi tạo
    queue = deque([start])
    visited = []

    while queue:
        vertex = queue.popleft()  # Lấy phần tử từ đầu hàng đợi

        seq.append(vertex)

        for neighbor, _ in adj_list.get(vertex, []):
            if neighbor not in visited:
                queue.append(neighbor)
                visited.append(neighbor)
    
    return seq

def all_components_bfs(adj_list, vertices):
    components = []
    _vertices = vertices.copy()

    while _vertices:
        component = bfs(adj_list, _vertices)
        components.append(component)

        # Xóa các đỉnh đã được duyệt khỏi danh sách đỉnh chưa duyệt
        for v in component:
            _vertices.remove(v)
    
    return components

def find_path_bfs(adj_list, vertices, start, end):
    if start not in adj_list or end not in adj_list:
        return None
    
    queue = deque([start])
    visited = [start]
    parent = {start: None}

    while queue:
        vertex = queue.popleft()  # Lấy phần tử từ đầu hàng đợi

        if vertex == end:
            path = []
            while vertex is not None:
                path.append(vertex)
                vertex = parent[vertex]
            path.reverse()
            return path  

        for neighbor, _ in adj_list.get(vertex, []):
            if neighbor not in visited:
                parent[neighbor] = vertex # Ghi lại cha của neighbor
                queue.append(neighbor)
                visited.append(neighbor)

    return None
##################################################################################################################################
#Tu start đến tất cả các đỉnh
#Dung cho đồ thị có trọng số(ko âm)
def dijkstra(adj_list, vertices, start = None):
    if start is None:
        start = vertices[0]

    distances = {v: float('inf') for v in vertices}
    distances[start] = 0
    pred = {v: [] for v in vertices}

    priority_queue = [(0, start)]  # (khoảng cách, đỉnh)

    while priority_queue:
        curr_distance, curr_vertex = heapq.heappop(priority_queue)

        if curr_distance > distances[curr_vertex]:
            continue

        for neighbor, weight in adj_list[curr_vertex]:
            distance = curr_distance + weight

            # Nếu tìm thấy khoảng cách ngắn hơn, cập nhật
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                pred[neighbor] = pred[curr_vertex] + [curr_vertex]
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances, pred

def bellman_ford(vertices, edges, start =None):
    if start is None:
        start = vertices[0]
    # Khởi tạo khoảng cách từ nguồn đến các đỉnh
    distance = {vertex: float('inf') for vertex in vertices}
    distance[start] = 0
    pred = {v: set() for v in vertices}

    # Lặp qua tất cả các cạnh (n-1 lần)
    for _ in range(len(vertices) - 1):
        for u, v, weight in edges:
            if distance[u] != float('inf') and distance[u] + weight < distance[v]:
                distance[v] = distance[u] + weight

    # Kiểm tra chu trình âm
    for u, v, weight in edges:
        if distance[u] != float('inf') and distance[u] + weight < distance[v]:
            raise ValueError("Đồ thị có chu trình âm")

    return distance

import heapq

def johnson(vertices, edges):
    # Bước 1: Thêm đỉnh giả "s" kết nối đến tất cả các đỉnh trong đồ thị
    new_vertex = "s"
    vertices.append(new_vertex)
    new_edges = edges + [(new_vertex, v, 0) for v in vertices if v != new_vertex]

    # Bước 2: Chạy Bellman-Ford từ đỉnh "s"
    try:
        h = bellman_ford(vertices, new_edges, new_vertex)
    except ValueError as e:
        raise ValueError("Đồ thị có chu trình âm nên không thể áp dụng thuật toán Johnson")

    # Bước 3: Tính trọng số mới (w') cho các cạnh
    reweighted_edges = []
    adj_list = {v: [] for v in vertices}
    for u, v, w in edges:
        new_weight = w + h[u] - h[v]
        reweighted_edges.append((u, v, new_weight))
        adj_list[u].append((v, new_weight))

    # Bước 4: Loại bỏ đỉnh "s" khỏi đồ thị
    vertices.remove(new_vertex)
    del adj_list[new_vertex]

    # Bước 5: Chạy Dijkstra cho từng đỉnh
    all_pairs_shortest_paths = {}
    for u in vertices:
        distances, _ = dijkstra(adj_list, vertices, u)
        # Khôi phục khoảng cách gốc
        for v in distances:
            distances[v] = distances[v] - h[u] + h[v]
        all_pairs_shortest_paths[u] = distances

    return all_pairs_shortest_paths


#########CÂY KHUNG NHO NHAT###############
#Prim chạy trên một thành phần liên thông
def prim(adj_list, vertices):
    # Khởi tạo cây khung mini (MST) và tập đỉnh đã chọn
    MST = []
    visited = set()

    start_vertex = vertices[0]
    visited.add(start_vertex)

    #Tạo tập cạnh ứng viên (các cạnh có 1 đỉnh thuộc MST)
    candidate_edges = []
    for neighbor, weight in adj_list[start_vertex]:
        heapq.heappush(candidate_edges, (weight, start_vertex, neighbor))

    # Tiếp tục cho đến khi MST có (n-1) cạnh
    while len(MST) < len(vertices) - 1:
        # Chọn cạnh trọng số nhỏ nhất
        weight, u, v = heapq.heappop(candidate_edges)

        # Kiểm tra nếu đỉnh đích v chưa thuộc MST
        if v not in visited:
            # Thêm cạnh vào MST
            MST.append((u, v, weight))
            visited.add(v)

            for neighbor, edge_weight in adj_list[v]:
                if neighbor not in visited:
                    heapq.heappush(candidate_edges, (edge_weight, v, neighbor))

    total_weight = 0
    for _,_,w in MST:
        total_weight += w

    return MST,total_weight

def find(parent, x):
        if parent[x] != x:
            parent[x] = find(parent, parent[x])  # Path compression
        return parent[x]

    # Hàm gộp hai đỉnh vào cùng một tập hợp con
def union(parent, rank, x, y):
    rootX = find(parent, x)
    rootY = find(parent, y)

    if rootX != rootY:
        # Union by rank
        if rank[rootX] > rank[rootY]:
            parent[rootY] = rootX
        elif rank[rootX] < rank[rootY]:
            parent[rootX] = rootY
        else:
            parent[rootY] = rootX
            rank[rootX] += 1

def boruvka(vertices, edges):
    # Khởi tạo tập hợp các thành phần ban đầu (mỗi đỉnh là một thành phần riêng)
    parent = {v: v for v in vertices}
    rank = {v: 0 for v in vertices}

    mst = []  # Danh sách lưu các cạnh của cây khung nhỏ nhất
    total_weight = 0  # Tổng trọng số của cây khung nhỏ nhất

    while len(set(find(parent, v) for v in vertices)) > 1:  # Lặp đến khi chỉ còn 1 thành phần
        # Bước 1: Tìm cạnh nhỏ nhất nối từ mỗi thành phần
        cheapest = {find(parent, v): None for v in vertices}  # Lưu cạnh nhỏ nhất cho mỗi thành phần

        for u, v, weight in edges:
            root_u = find(parent, u)
            root_v = find(parent, v)
            if root_u != root_v:  # Chỉ xét cạnh nối hai thành phần khác nhau
                # Nếu cạnh này là cạnh nhỏ nhất của root_u hoặc root_v
                if cheapest[root_u] is None or cheapest[root_u][2] > weight:
                    cheapest[root_u] = (u, v, weight)
                if cheapest[root_v] is None or cheapest[root_v][2] > weight:
                    cheapest[root_v] = (u, v, weight)

        # Bước 2: Thêm các cạnh nhỏ nhất vào MST và hợp nhất thành phần
        for edge in cheapest.values():
            if edge is not None:
                u, v, weight = edge
                root_u = find(parent, u)
                root_v = find(parent, v)
                if root_u != root_v:  # Đảm bảo không tạo chu trình
                    union(parent, rank, u, v)
                    mst.append((u, v, weight))
                    total_weight += weight

    return mst, total_weight


def kruskal(vertices, edges):    
    edges = sorted(edges, key= lambda x: x[2])

    parent = {v: v for v in vertices}  # Mảng parent để lưu đại diện các đỉnh
    rank = {v: 0 for v in vertices}  # Mảng rank để tối ưu việc gộp các tập hợp con

    mst = []  # Danh sách lưu các cạnh của cây khung nhỏ nhất
    total_weight = 0  # Tổng trọng số của cây khung nhỏ nhất

    for u, v, weight in edges:
        # Nếu hai đỉnh không thuộc cùng một tập hợp con, kết nối chúng
        if find(parent, u) != find(parent, v):
            union(parent, rank, u, v)
            mst.append((u, v, weight))  # Thêm cạnh vào MST
            total_weight += weight

    return mst, total_weight

#Tìm chu trình Euler
def hierholzer(adj_list, vertices):
    # Kiểm tra điều kiện 
    for vertex in vertices:
        if len(adj_list[vertex]) % 2 != 0:
            return "Không tồn tại chu trình Euler: Đỉnh có bậc lẻ."
        
    local_adj_list = copy.deepcopy(adj_list)
    start_vertex = vertices[0]
    stack = [start_vertex]  # Stack dùng để lưu các đỉnh trong quá trình duyệt
    euler_path = []         # Kết quả: Chu trình Euler

    while stack:
        current_vertex = stack[-1]

        if local_adj_list[current_vertex]:
            next_vertex, weight = local_adj_list[current_vertex].pop()  # Lấy đỉnh kề và trọng số
            local_adj_list[next_vertex].remove((current_vertex, weight))
            stack.append(next_vertex)
        else:
            # Nếu không còn cạnh nào, di chuyển ngược lại
            euler_path.append(stack.pop())

    # Đảo ngược chu trình Euler
    euler_path.reverse()
    return euler_path

####
def fleury(adj_list, vertices):
    # Kiểm tra điều kiện 
    for vertex in vertices:
        if len(adj_list[vertex]) % 2 != 0:
            return "Không tồn tại chu trình Euler: Đỉnh có bậc lẻ."

    path = []
    current_vertex = vertices[0]  # Bắt đầu từ đỉnh đầu tiên
    local_adj_list = copy.deepcopy(adj_list)
    
    while sum(len(local_adj_list[vertex]) for vertex in vertices) > 0:  # Còn cạnh chưa được đi
        for neighbor,_ in local_adj_list[current_vertex]:
            if is_bridges(local_adj_list,vertices, current_vertex,neighbor) or len(local_adj_list[current_vertex]) == 1:
                path.append((current_vertex, neighbor))
                local_adj_list[current_vertex] = [pair for pair in local_adj_list[current_vertex] if pair[0] != neighbor]
                local_adj_list[neighbor] = [pair for pair in local_adj_list[neighbor] if pair[0] != current_vertex]
                current_vertex = neighbor
                break
    
    return path

def is_bridges(adj_list,vertices,v_from,v_to):
    local_adj_list = copy.deepcopy(adj_list)
    local_adj_list[v_from] = [pair for pair in local_adj_list[v_from] if pair[0] != v_to]
    local_adj_list[v_to] = [pair for pair in local_adj_list[v_to] if pair[0] != v_from]
    if len(all_components_dfs(local_adj_list,vertices)) > 1:
        return False
    return True


#tìm các đỉnh cắt đơn, không phải tập đỉnh cắt tối thiểu
def CriticalVertices(adj_list,vertices):
    criticalVertices = []

    components = len(all_components_dfs(adj_list,vertices))

    for vertex in vertices:

        _vertices = list(set(vertices) - set([vertex]))
        new_adj_list = {v: [neigh for neigh in adj_list[v] if neigh != vertex] for v in adj_list}
        newComponents = len(all_components_dfs(new_adj_list,_vertices))

        if newComponents > components:
            criticalVertices.append(vertex)
    return criticalVertices
    
#Tìm cạnh cầu
def Bridges(adj_list, vertices):
    bridges = []
    components = len(all_components_dfs(adj_list, vertices))

    edges = []
    for v_from in adj_list:
        for v_to, _ in adj_list[v_from]:
            if (v_to, v_from) not in edges:  # Tránh thêm cạnh ngược (đồ thị vô hướng)
                edges.append((v_from, v_to))

    for v_from, v_to in edges:
        local_adj_list = copy.deepcopy(adj_list)

        # Xóa cạnh giữa v_from và v_to
        local_adj_list[v_from] = [pair for pair in local_adj_list[v_from] if pair[0] != v_to]
        local_adj_list[v_to] = [pair for pair in local_adj_list[v_to] if pair[0] != v_from]

        new_components = len(all_components_dfs(local_adj_list, vertices)) 

        if new_components > components:
            bridges.append((v_from, v_to))

    return bridges

#Chu trinh haminton
def bellman_held_karp(adj_list, vertices):
    adj_df = Database.adj_list_to_adj_df(adj_list, vertices)
    n = len(vertices)
    dp = {}
    parent = {}  # Dùng để dò ngược
    
    # Khởi tạo: g({k}, k) = d(1, k)
    for k in range(1, n):
        dp[(frozenset([k]), k)] = adj_df.iloc[0, k]
        parent[(frozenset([k]), k)] = 0  # Xuất phát từ đỉnh đầu tiên

    # Xây dựng tập hợp con (subset) kích thước tăng dần 
    for subset_size in range(2, n): 
        for subset in itertools.combinations(range(1, n), subset_size):
            subset = frozenset(subset)
            for k in subset:
                # Tính g(S, k) = min [g(S \ {k}, m) + d(m, k)]
                subset_without_k = subset - {k}
                min_cost, min_m = min(
                    ((dp[(subset_without_k, m)] + adj_df.iloc[m, k]), m)
                    for m in subset_without_k
                )
                dp[(subset, k)] = min_cost
                parent[(subset, k)] = min_m  # Lưu đỉnh m tốt nhất dẫn đến k

    # Tính giá trị tối ưu: opt = min [g({2, ..., n}, k) + d(k, 1)]
    full_set = frozenset(range(1, n))
    opt, last_node = min(
        ((dp[(full_set, k)] + adj_df.iloc[k, 0]), k)
        for k in range(1, n)
    )

    if opt == np.inf:
        return "Không có chu trình Hamilton", []

    # Truy vết để tìm chu trình
    path = [0]  # Bắt đầu từ đỉnh 0
    current_set = full_set
    current_node = last_node

    while current_node != 0:
        path.append(current_node)
        next_node = parent[(current_set, current_node)]
        current_set = current_set - {current_node}
        current_node = next_node

    path.append(0)  # Kết thúc tại đỉnh 0
    path = [vertices[i] for i in path]
    path.reverse()
    return float(opt), path

def branch_and_bound(adj_list, vertices):
    n = len(vertices)
    best_path = None
    best_cost = float('inf')
    
    # Khởi tạo priority queue
    priority_queue = []
    for start in vertices:
        # Mỗi phần tử trong hàng đợi là (cost, path, visited_dict)
        heapq.heappush(priority_queue, (0, [start], {v: v == start for v in vertices}))
    
    while priority_queue:
        current_cost, path, visited = heapq.heappop(priority_queue)
        
        # Nếu tất cả các đỉnh đã được thăm
        if len(path) == n:
            last_vertex = path[-1]
            first_vertex = path[0]
            # Kiểm tra cạnh quay về đỉnh đầu
            for neighbor, weight in adj_list[last_vertex]:
                if neighbor == first_vertex:
                    cycle_cost = current_cost + weight
                    if cycle_cost < best_cost:
                        best_cost = cycle_cost
                        best_path = path + [first_vertex]
            continue

        # Duyệt qua các đỉnh kề
        current_vertex = path[-1]
        for neighbor, weight in adj_list[current_vertex]:
            if not visited[neighbor]:
                # Cập nhật trạng thái
                new_cost = current_cost + weight
                new_path = path + [neighbor]
                new_visited = visited.copy()
                new_visited[neighbor] = True

                # Thêm vào hàng đợi
                heapq.heappush(priority_queue, (new_cost, new_path, new_visited))
    
    return (float(best_cost), best_path) if best_path else (None, None)

###TÔ MÀU#######
def greedy_coloring(vertices, adj_list):
    # Khởi tạo một từ điển để lưu màu của các đỉnh, ban đầu chưa có màu (None)
    colorcollection = ['yellow', 'orange', 'blue','Green']
    colors = {vertex: None for vertex in vertices}

    # Duyệt qua tất cả các đỉnh trong đồ thị
    for vertex in vertices:
        # Lấy các màu đã được sử dụng cho các đỉnh kề của đỉnh hiện tại
        neighbor_colors = set()
        for neighbor,_ in adj_list[vertex]:
            if colors[neighbor] is not None:
                neighbor_colors.add(colors[neighbor])

        colorapply = None
        for color in colorcollection:
            if color not in neighbor_colors:
                colorapply = color
                break

        # Gán màu cho đỉnh
        colors[vertex] = colorapply

    return colors