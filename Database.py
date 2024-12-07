import pandas as pd
import numpy as np
from neo4j import GraphDatabase

def Connect():
    NEO4J_URI = "neo4j+s://45b7324c.databases.neo4j.io" 
    USERNAME = "neo4j"
    PASSWORD = "5l5KqmekdQxJ9vSz5sR8dQeGciU-rFBk61pt8b47IYk" 

    driver = GraphDatabase.driver(NEO4J_URI, auth=(USERNAME, PASSWORD))
    return driver

def all_graph_data(driver):
    with driver.session() as session:
        # Truy vấn Cypher
        query = """
        MATCH (g:Graph)-[:CONTAINS]->(n:Node)
        RETURN g.id AS graph_id, g.name AS graph_name, COUNT(n) AS node_count
        """
        results = session.run(query)
        
        # In dữ liệu
        for record in results:
            print(f"ID: {record['graph_id']}, Tên: {record['graph_name']}, Số đỉnh: {record['node_count']}")


#Load về dưới dạng danh sách kề
def Load_Graph(driver, graph_id):

    with driver.session() as session:
        # Truy vấn các cạnh của đồ thị
        query = """
        MATCH (g:Graph {id: $graph_id})-[:CONTAINS]->(n1:Node)
        MATCH (n1)-[r:CONNECTED]->(n2:Node)
        RETURN n1.name AS from_node, n2.name AS to_node, r.weight AS weight
        """
        results = session.run(query, graph_id=graph_id)
        
        # Xây dựng danh sách kề
        adj_list = {}
        vertices = set()
        edges =[]
        for record in results:
            from_node = record["from_node"]
            to_node = record["to_node"]
            weight = record["weight"]
            
            vertices.add(from_node)
            vertices.add(to_node)

            # Thêm vào danh sách kề
            if from_node not in adj_list:
                adj_list[from_node] = []
            adj_list[from_node].append((to_node, weight))

            edges.append((from_node, to_node, weight))
        
        vertices = sorted(vertices)
        return adj_list, vertices,edges
    
def adj_list_to_edges_list(adj_list, vertices):
    edgesList = []

    for v_from in vertices:
        for v_to, weight in adj_list[v_from]:  # Lấy cặp đỉnh v_from và v_to
            edgesList.append((v_from, v_to,weight))  # Thêm cặp đỉnh vào edgesList
    
    return edgesList

def adj_list_to_adj_df(adj_list, vertices):
    df = pd.DataFrame(np.inf, index=vertices, columns=vertices)
    
    # Đặt đường chéo là 0
    for vertex in vertices:
        df.at[vertex, vertex] = 0

    for vertex, neighbors in adj_list.items():
        for neighbor, weight in neighbors:
            df.at[vertex, neighbor] = weight
    return df
