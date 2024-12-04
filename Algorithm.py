#Thu vien
from collections import deque
import pandas as pd
import numpy as np
import heapq
import copy

##################################################################################################################################
def dfs(adj_list, vertices, start=None):
    seq = []
    if start is None:
        start = vertices[0]
    # Khởi tạo 
    stack = [start]
    visited = []

    while stack:
        vertex = stack.pop()
        if vertex in visited:
            continue

        visited.append(vertex)
        seq.append(vertex)

        for neighbor, _ in adj_list.get(vertex, []):
            if neighbor not in visited and neighbor in vertices:
                stack.append(neighbor)
    
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
    visited = []
    parent = {start: None}

    while stack:
        vertex = stack.pop()

        if vertex in visited:
            continue
        visited.append(vertex)

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
def Bridges(adj_list, vertices, edges):
    bridges = []
    
    # Sử dụng bản sao đồ thị để kiểm tra số thành phần liên thông ban đầu
    components = len(all_components_dfs(adj_list, vertices))

    for v_from, v_to, _ in edges:
        local_adj_list = copy.deepcopy(adj_list)  # Sao chép đồ thị
        
        # Xóa cạnh giữa v_from và v_to trong danh sách kề của v_from
        if v_from in local_adj_list:
            local_adj_list[v_from].remove((v_to, _))  # Xóa v_to khỏi danh sách kề của v_from
            print(v_to)
        # Xóa cạnh giữa v_to và v_from trong danh sách kề của v_to
        if v_to in local_adj_list:
            local_adj_list[v_to].remove((v_from, _))  # Xóa v_from khỏi danh sách kề của v_to
            print(v_from)
        # Kiểm tra số thành phần liên thông sau khi xóa cạnh
        newComponents = len(all_components_dfs(local_adj_list, vertices))  # Sử dụng local_adj_list đã thay đổi
        
        # Nếu số thành phần liên thông tăng, đây là một cầu (bridge)
        if newComponents > components:
            bridges.append((v_from, v_to))  # Thêm cặp (v_from, v_to) vào danh sách bridges
    
    return bridges

