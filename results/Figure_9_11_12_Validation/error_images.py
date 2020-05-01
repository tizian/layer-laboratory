import os
import numpy as np
import scipy.sparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    import mitsuba
except ImportError as error:
    print(error)
    print("Could not import the Mitsuba 2 python modules. Make sure to \"source setpath.sh\" before running this script.")

import mitsuba
mitsuba.set_variant('scalar_rgb')
import mitsuba.layer
from mitsuba.core import Bitmap, Struct

# Note: requires the vips image processing toolbox. E.g. brew install vips

# image, ref
add_cardboard_names = [
    'measured_addition/cardboard/layer_020_028_added',
    'measured_addition/cardboard/layer_020_028',
    'measured_addition/cardboard/layer_020_028_dE'
]

add_goldpaper_names = [
    'measured_addition/goldpaper/layer_020_004_added',
    'measured_addition/goldpaper/layer_020_004',
    'measured_addition/goldpaper/layer_020_004_dE'
]

sub_cardboard_names = [
    'measured_subtraction/cardboard/blue-cardboard_subtracted',
    'measured_subtraction/cardboard/blue-cardboard',
    'measured_subtraction/cardboard/blue-cardboard_subtracting_dE',
]

sub_goldpaper_names = [
    'measured_subtraction/goldpaper/goldpaper0_subtracted',
    'measured_subtraction/goldpaper/goldpaper0',
    'measured_subtraction/goldpaper/goldpaper0_subtracting_dE'
]

tonemap = []
for i in range(2):
    tonemap.append(add_cardboard_names[i])
    tonemap.append(add_goldpaper_names[i])
    tonemap.append(sub_cardboard_names[i])
    tonemap.append(sub_goldpaper_names[i])

for img in tonemap:
    path = img + ".exr"
    bitmap = Bitmap(path)
    bitmap.convert(Bitmap.PixelFormat.RGB, Struct.Type.UInt8, srgb_gamma=True).write(os.path.join("", path.replace('.exr', '.jpg')))

def create_dE(names):
    os.system('vips im_dE_fromdisp %s.jpg %s.jpg %s.png screen' % (names[0], names[1], names[2]))

create_dE(add_cardboard_names)
create_dE(add_goldpaper_names)
create_dE(sub_cardboard_names)
create_dE(sub_goldpaper_names)

def create_mapped_dE(names, maximum):
    in_name = names[2] + '.png'
    dE  = plt.imread(in_name)
    dE = (255*dE).astype(int)

    cmap = plt.cm.hot
    norm = mcolors.Normalize(vmin=0, vmax=maximum)

    output = cmap(norm(dE))

    out_name = names[2] + '.jpg'
    plt.imsave(out_name, output)
    # plt.imshow(output)
    # plt.show()

create_mapped_dE(add_cardboard_names, 10)
create_mapped_dE(add_goldpaper_names, 10)
create_mapped_dE(sub_cardboard_names, 20)
create_mapped_dE(sub_goldpaper_names, 20)

def create_colorbar(maximum, path):
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.05])

    cmap = plt.cm.hot
    norm = mcolors.Normalize(vmin=0, vmax=maximum)

    cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')

    plt.savefig('%s/colorbar_dE.pdf' % path, bbox_inches='tight')
    # plt.show()
    # plt.close()

create_colorbar(10, 'measured_addition')
create_colorbar(20, 'measured_subtraction')