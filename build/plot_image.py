import h5py
import matplotlib.pyplot as plt
import numpy as np


input_h5_path = 'tmp.h5'
data_name = '/group/dset1'

with h5py.File(input_h5_path, 'r') as f:
  data = f[data_name][:]


# print(data.flatten())
# print(data[:,:,2].flatten())

data = data.transpose(1, 0, 2)

# print(data[:,:,2].flatten())
print('data.shape:', data.shape)
print(data)
print(data.flatten())


plt.subplot(2, 3, 1)
plt.imshow(data, origin='lower')

plt.subplot(2, 3, 4)
plt.imshow(data[:, :, 0], origin='lower')
plt.colorbar()
plt.title(f'Channel 1, min={np.min(data[:, :, 0])}, max={np.max(data[:, :, 0])}')

plt.subplot(2, 3, 5)
plt.imshow(data[:, :, 1], origin='lower')
plt.colorbar()
plt.title(f'Channel 2, min={np.min(data[:, :, 1])}, max={np.max(data[:, :, 1])}')

plt.subplot(2, 3, 6)
plt.imshow(data[:, :, 2], origin='lower')
plt.colorbar()
plt.title(f'Channel 3, min={np.min(data[:, :, 2])}, max={np.max(data[:, :, 2])}')

plt.show()