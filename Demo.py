import sys
from pymongo import MongoClient
from bson.objectid import ObjectId

import Database
import Algorithm

try:
    collection = Database.Connect()
    #all_data = collection.find().sort("name",1)
except Exception as e:
    print("Kết nối thất bại!")
    sys.exit(1)
    

adj_list, vertices, edges = Database.Load_Graph(collection,'674df8fbb2c1143f2130cf15')
print(adj_list)

seq = Algorithm.all_components_dfs(adj_list, vertices)
print(seq)

# seq_dfs = Algorithm.dfs(adj_list,vertices,'v01')
# print("DFS: " + str(seq_dfs))

# path = Algorithm.find_path_dfs(adj_list,vertices,'v02','v01')
# print(path)

# seq_bfs = Algorithm.bfs(adj_list,vertices,'v02')
# print("BFS: " + str(seq_bfs))

# path = Algorithm.find_path_bfs(adj_list,vertices,'v02','v01')
# print(path)