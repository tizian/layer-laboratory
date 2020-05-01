import os
import numpy as np

scene = "dragon.xml"
spp = 32

# Anisotropic, analytic addition
print("Render anisotropic addition results ({} spp) ..".format(spp))
base_directory = "analytic_addition/"

names = [
    "aniso_addition_base",
    "aniso_addition_top",
    "aniso_addition_added"
]

for name in names:
    filepath = base_directory + name + ".bsdf"
    if not os.path.isfile(filepath):
        print("""The layered BSDF file \"{}.bsdf\" required for this rendering was not yet precomputed.
                 Please see the previous instructions in README.md regarding analytic addition (Figure 9a).""".format(filepath))

    print("")
    print("  - \"{}\" ..".format(filepath))
    outname = base_directory + name + ".exr"
    cmd = "mitsuba {} -o {} -Dspp={} -Dlayer_filename={} ".format(scene, outname, spp, filepath)
    os.system(cmd)
    print("")
print("done.\n")


# Anisotropic, analytic subtraction
print("Render anisotropic subtraction results ({} spp) ..".format(spp))
base_directory = "analytic_subtraction/"

names = [
    "aniso_subtraction_added",
    "aniso_subtraction_top",
    "aniso_subtraction_subtracted",
    "aniso_subtraction_ref"
]

for name in names:
    filepath = base_directory + name + ".bsdf"
    if not os.path.isfile(filepath):
        print("""The layered BSDF file \"{}.bsdf\" required for this rendering was not yet precomputed.
                 Please see the previous instructions in README.md regarding analytic subtraction (Figure 9b).""".format(filepath))

    print("")
    print("  - \"{}\" ..".format(filepath))
    outname = base_directory + name + ".exr"
    cmd = "mitsuba {} -o {} -Dspp={} -Dlayer_filename={} ".format(scene, outname, spp, filepath)
    os.system(cmd)
    print("")
print("done.\n")




envmap_scale = 2.5

# Measured addition
print("Render measured addition results ({} spp) ..".format(spp))
base_directory = "measured_addition/"

names = [
    "cardboard/blue-cardboard",
    "cardboard/layer_020_028_added",
    "cardboard/layer_020_028",
    "cardboard/matte-film-01",
    "goldpaper/goldpaper0",
    "goldpaper/layer_020_004_added",
    "goldpaper/layer_020_004",
    "goldpaper/matte-film-01"
]

for name in names:
    filepath = base_directory + name + ".bsdf"
    if not os.path.isfile(filepath):
        print("""The layered BSDF file \"{}.bsdf\" required for this rendering was not yet precomputed.
                 Please see the previous instructions in README.md regarding measured addition (Figure 9c,e).""".format(filepath))

    print("")
    print("  - \"{}\" ..".format(filepath))
    outname = base_directory + name + ".exr"
    cmd = "mitsuba {} -o {} -Dspp={} -Dlayer_filename={} -Denvmap_scale={} ".format(scene, outname, spp, filepath, envmap_scale)
    os.system(cmd)
    print("")
print("done.\n")


# Measured subtraction
print("Render measured subtraction results ({} spp) ..".format(spp))
base_directory = "measured_subtraction/"

names = [
    "cardboard/blue-cardboard_subtracted",
    "cardboard/blue-cardboard",
    "cardboard/layer_020_028",
    "cardboard/matte-film-01",
    "goldpaper/goldpaper0_subtracted",
    "goldpaper/goldpaper0",
    "goldpaper/layer_020_004",
    "goldpaper/matte-film-01"
]

for name in names:
    filepath = base_directory + name + ".bsdf"
    if not os.path.isfile(filepath):
        print("""The layered BSDF file \"{}.bsdf\" required for this rendering was not yet precomputed.
                 Please see the previous instructions in README.md regarding measured subtraction (Figure 9d,f).""".format(filepath))

    print("")
    print("  - \"{}\" ..".format(filepath))
    outname = base_directory + name + ".exr"
    cmd = "mitsuba {} -o {} -Dspp={} -Dlayer_filename={} -Denvmap_scale={} ".format(scene, outname, spp, filepath, envmap_scale)
    os.system(cmd)
    print("")
print("done.\n")
