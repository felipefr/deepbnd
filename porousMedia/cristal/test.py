import matplotlib.pyplot as plt
import microstructpy as msp
import scipy
import numpy as np

# <?xml version="1.0" encoding="UTF-8"?>
# <input>
#     <material>
#         <name> Matrix </name>
#         <material_type> matrix </material_type>
#         <fraction> 2 </fraction>
#         <shape> circle </shape>
#         <size>
#             <dist_type> uniform </dist_type>
#             <loc> 0 </loc>
#             <scale> 1.5 </scale>
#         </size>
#     </material>

#     <material>
#         <name> Inclusions </name>
#         <fraction> 1 </fraction>
#         <shape> circle </shape>
#         <diameter> 2 </diameter>
#     </material>

#     <domain>
#         <shape> square </shape>
#         <side_length> 20 </side_length>
#         <corner> (0, 0) </corner>
#     </domain>

#     <settings>
#         <verbose> True </verbose>
#         <directory> intro_1_basic </directory>
#     </settings>
# </input>


domain = msp.geometry.Square()

# param = {'shape': 'circle', 'size': 0.10}


param = {'material1': {'material_type' : 'matrix',
         'shape' : 'circle',
         'size' : scipy.stats.uniform(scale=0.2)},
         'material2': {'fraction' : 1,
         'shape' : 'circle',
         'size' : 0.2}
         }

# 'size' : scipy.stats.lognorm(scale=0.1, s=0.2)

param_a = np.array([0.5 * domain.area,0.5 * domain.area])
seeds = msp.seeding.SeedList.from_info(param, param_a)
# inds = np.flip(np.argsort([s.volume for s in foam_seeds]))
# foam_seeds = foam_seeds[inds]

# Position seeds in domain
seeds.position(domain)

# Create polygonal mesh
polygon_mesh = msp.meshing.PolyMesh.from_seeds(seeds, domain,   )

# Create triangular mesh
triangle_mesh = msp.meshing.TriMesh.from_polymesh(polygon_mesh, mesh_size = 0.001)

# Plot outputs
for output in [seeds, polygon_mesh, triangle_mesh]:
    plt.figure()
    output.plot(edgecolor='k')
    plt.axis('image')
    plt.axis([-0.5, 0.5, -0.5, 0.5])
    # plt.show()
    
    
