import spectral
import numpy as np

def load_msi_cube(hdr_path):

    img = spectral.open_image(str(hdr_path))

    cube = img.load()

    cube = np.array(cube)

    return cube