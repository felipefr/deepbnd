import matplotlib.pyplot as plt
import microstructpy as msp
import scipy


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
#         <directory> intro_2_quality </directory>
#         <verbose> True </verbose>

#         <!-- Mesh Quality Settings -->
#         <mesh_min_angle> 25 </mesh_min_angle>
#         <mesh_max_volume> 1 </mesh_max_volume>
#         <mesh_max_edge_length> 0.1 </mesh_max_edge_length>
#     </settings>
# </input>

min_angle = 25
max_volume = 1
max_edge_length = 0.1

domain = msp.geometry.Square()

# param = {'shape': 'circle', 'size': 0.10}


param1 = {'material_type': 'matrix', 'shape': 'circle', 'fraction': 1,
            'size': scipy.stats.lognorm(scale=0.1, s=0.2)}

param2 = {'material_type': 'Inclusions', 'shape': 'circle', 'fraction': 2,
            'size': scipy.stats.lognorm(scale=0.1, s=0.2)}


param_a = 0.5 * domain.area
seeds = msp.seeding.SeedList.from_info([param1, param2], param_a)


# inds = np.flip(np.argsort([s.volume for s in foam_seeds]))
# foam_seeds = foam_seeds[inds]

# Position seeds in domain
seeds.position(domain)

# Create polygonal mesh
polygon_mesh = msp.meshing.PolyMesh.from_seeds(seeds, domain,   )

# Create triangular mesh
triangle_mesh = msp.meshing.TriMesh.from_polymesh(polygon_mesh, mesh_size = 0.001, 
                                                  max_volume = max_volume, min_angle = min_angle, max_edge_length= max_edge_length)

# Plot outputs
for output in [seeds, polygon_mesh, triangle_mesh]:
    plt.figure()
    output.plot(edgecolor='k')
    plt.axis('image')
    plt.axis([-0.5, 0.5, -0.5, 0.5])
    # plt.show()
    
    
