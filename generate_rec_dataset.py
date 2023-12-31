import meshio
import numpy as np


triangles=meshio.read("data/bunny_dense_test_0.ply").cells_dict["triangle"]
all_points_gen=np.load("all_points_gen.npy")
for i in range(600):
    points=all_points_gen[i]
    points=points-np.mean(points,axis=0)
    meshio.write_points_cells("data/bunny_dense_gen_"+str(i)+".ply",points,{"triangle":triangles})
