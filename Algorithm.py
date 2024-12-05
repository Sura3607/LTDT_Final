#Thu vien
from collections import deque
import pandas as pd
import numpy as np
import heapq
import itertools
import copy
import Database

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
                visited.append(vertex)
    
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
                visited.append(vertex)
    
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

#Tìm chu trình Euler
def hierholzer(adj_list, vertices):
    local_adj_list = copy.deepcopy(adj_list)
    
    # Kiểm tra điều kiện 
    for vertex in vertices:
        if len(local_adj_list[vertex]) % 2 != 0:
            return "Không tồn tại chu trình Euler: Đỉnh có bậc lẻ."

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

#Hàm tạo tập hợp
def generate_subsets(cut_sets):
    subsets = []

    def backtrack(start, path):
        # Thêm tập con hiện tại vào danh sách
        subsets.append(path)

        for i in range(start, len(cut_sets)):
            # Gọi đệ quy với đỉnh tiếp theo
            backtrack(i + 1, path + [cut_sets[i]])

    backtrack(0, [])
    subsets.sort(key=len)
    return subsets

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

def held_karp_tsp(adj_list, vertices):
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
    return opt, path
