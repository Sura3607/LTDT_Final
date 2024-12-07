import sys

import Database
import Algorithm

try:
    driver = Database.Connect()
    print("Kết nối thành công!")
except Exception as e:
    print("Kết nối thất bại!")
    sys.exit(1)
    

adj_list, vertices,edges= Database.Load_Graph(driver,'673616adf36b60bb91cb64a0')
print(adj_list)
# print(vertices)
# print(edges)

# print(Algorithm.all_components_dfs(adj_list,vertices))

#To mau
# print(Algorithm.greedy_coloring(vertices,adj_list))

#Haminton
# print(Algorithm.bellman_held_karp(adj_list,vertices))
# print(Algorithm.branch_and_bound(adj_list, vertices))

#Duong di ngan nhat
# print(Algorithm.dijkstra(adj_list,vertices))
# print(Algorithm.bellman_ford(vertices,edges))
# print(Algorithm.johnson(vertices,edges))

#Cay khung
# print(Algorithm.prim(adj_list,vertices))
# print(Algorithm.kruskal(vertices,edges))
# print(Algorithm.boruvka(adj_list,edges))

#Dinh cat
# print(Algorithm.CriticalVertices(adj_list,vertices))

#Canh cat
# print(Algorithm.Bridges(adj_list,vertices))

#Euler
# print(Algorithm.hierholzer(adj_list,vertices))
# print(Algorithm.fleury(adj_list,vertices))