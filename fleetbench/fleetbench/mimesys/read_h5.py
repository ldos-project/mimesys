import h5py
import sys

if len(sys.argv) < 2:
    print("Usage: python read_h5.py <filename>")
    sys.exit(1)

filename = sys.argv[1]

with h5py.File(filename, 'r') as f:
    print("Keys in HDF5 file:", list(f.keys()))

    tensor_ds = f['execution_plan']  # Change 'tensor' if different dataset name

    print("Tensor shape:", tensor_ds.shape)
    print("Tensor dtype:", tensor_ds.dtype)

    tensor = tensor_ds[()]
    print("Tensor data:\n", tensor)

