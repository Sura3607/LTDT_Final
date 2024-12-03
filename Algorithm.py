#Thu vien
from collections import deque
import pandas as pd
import numpy as np
import heapq

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
            if neighbor not in visited:
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