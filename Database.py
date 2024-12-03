from pymongo import MongoClient
from bson.objectid import ObjectId

def Connect():
    connection_string = "mongodb+srv://ltdtFinal:z1PuIwaKQZvxkub8@graph-database-cluster.3976d.mongodb.net/?retryWrites=true&w=majority&appName=graph-database-cluster"
    # Kết nối đến MongoDB Atlas
    client = MongoClient(connection_string)
    # Lấy cơ sở dữ liệu và collection
    db = client["Graphs"]
    collection = db["graph"]
    print("Kết nối thành công đến MongoDB!")
    return collection


def Load_Graph(collection, graph_id):
    # Truy vấn dữ liệu theo _id
    graph_data = collection.find_one({"_id": ObjectId(graph_id)})

    if graph_data:
        directed = 'directed' in graph_data.get('tags', [])
        edges = graph_data["edges"] #Lấy cạnh
        adjacency_list = {}

        for edge in edges:
            v_from = edge["v_from"]
            v_to = edge["v_to"]
            weight = edge["weight"]

            # Thêm cạnh vào danh sách kề
            if v_from not in adjacency_list:
                adjacency_list[v_from] = []
            adjacency_list[v_from].append((v_to, weight))

            # Nếu là đồ thị vô hướng, thêm cạnh ngược lại
            if not directed:
                if v_to not in adjacency_list:
                    adjacency_list[v_to] = []
                adjacency_list[v_to].append((v_from, weight))

        vertices = set() #Tất cả các đỉnh
        for edge in edges:
            vertices.add(edge["v_from"])
            vertices.add(edge["v_to"])

        # Thêm các đỉnh cô lập vào danh sách kề
        for vertex in vertices:
            if vertex not in adjacency_list:
                adjacency_list[vertex] = []

        sorted_vertices = sorted(vertices, key=lambda x: int(x[1:]) if x[1:].isdigit() else x)

        return adjacency_list, sorted_vertices, edges
    else:
        print("Không tìm thấy đồ thị với _id đã cho.")
        return None, None, None
    