import torch
import ttnn

"""
This is a sample program to test to see if your ttnn library is working
with your Tenstorrent hardware as expected.
"""

device = ttnn.open_device(device_id=0)

a = torch.tensor([3, 3])
b = torch.tensor([2, 5])

a = ttnn.from_torch(a, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
b = ttnn.from_torch(b, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

output = a + b

assert [output[0], output[1]] == [5.0000, 8.0000]
assert output.shape == [2]
assert output.dtype == ttnn.bfloat16

print(output)

ttnn.close_device(device)