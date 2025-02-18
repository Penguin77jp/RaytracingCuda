import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt

def main():
    cmap = None
    args = argparse.ArgumentParser()
    args.add_argument('-i', help='Input file', required=True)
    args = args.parse_args()

    input_h5_file = args.i

    with h5py.File(input_h5_file, 'r') as f:
        image = f['/group/dset1'][:]
        image = np.transpose(image, (1, 0, 2))
    
    plt.subplot(2, 3, 1)
    plt.imshow(image, origin='lower')

    plt.subplot(2, 3, 4)
    plt.imshow(image[:, :, 0], cmap=cmap, origin='lower')
    plt.colorbar()

    plt.subplot(2, 3, 5)
    plt.imshow(image[:, :, 1], cmap=cmap, origin='lower')
    plt.colorbar()

    plt.subplot(2, 3, 6)
    plt.imshow(image[:, :, 2], cmap=cmap, origin='lower')
    plt.colorbar()

    plt.show()


if __name__ == '__main__':
    main()