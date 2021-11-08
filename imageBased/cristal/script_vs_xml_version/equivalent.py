import matplotlib.pyplot as plt
import microstructpy as msp
import scipy.stats

# Create Materials
material_1 = {
    'name': 'Matrix',
    'material_type': 'matrix',
    'fraction': 2,
    'shape': 'circle',
    'size': scipy.stats.uniform(loc=0, scale=1.5)
}

material_2 = {
    'name': 'Inclusions',
    'fraction': 1,
    'shape': 'circle',
    'diameter': 2
}

materials = [material_1, material_2]

# Create Domain
domain = msp.geometry.Square(side_length=15, corner=(0, 0))

# Create List of Un-Positioned Seeds
seed_area = domain.area
rng_seeds = {'size': 1}
seeds = msp.seeding.SeedList.from_info(materials,
                                       seed_area,
                                       rng_seeds)

# Position Seeds in Domain
seeds.position(domain)

# Create Polygonal Mesh
pmesh = msp.meshing.PolyMesh.from_seeds(seeds, domain)

# Create Triangular Mesh
# min_angle = 25
# tmesh = msp.meshing.TriMesh.from_polymesh(pmesh,
#                                           materials,
#                                           min_angle)

# Save txt files
seeds.write('seeds.txt')
pmesh.write('polymesh.txt')
# tmesh.write('trimesh.txt')

# Plot outputs
seed_colors = ['C' + str(s.phase) for s in seeds]
seeds.plot(facecolors=seed_colors, edgecolor='k')
plt.axis('image')
plt.savefig('seeds.png')
plt.clf()

poly_colors = [seed_colors[n] for n in pmesh.seed_numbers]
pmesh.plot(facecolors=poly_colors, edgecolor='k')
plt.axis('image')
plt.savefig('polymesh.png')
plt.clf()

# tri_colors = [seed_colors[n] for n in tmesh.element_attributes]
# tmesh.plot(facecolors=tri_colors, edgecolor='k')
# plt.axis('image')
# plt.savefig('trimesh.png')
# plt.clf()