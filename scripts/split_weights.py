import re

with open('modulos_verilog/weights_all.mif', 'r') as f:
    lines = f.readlines()

out_lines = []
for line in lines:
    m = re.match(r'^\s*\d+\s*:\s*([0-9a-fA-F]+)\s*;', line)
    if m:
        out_lines.append(m.group(1))

CONV_FILTERS = 4
CONV_KERNEL = 9
DENSE_SIZE = 900
DENSE_CLASSES = 7

# Write conv weights and biases (36 + 4 = 40)
with open('modulos_verilog/weights_conv.hex', 'w') as f:
    for val in out_lines[0:40]:
        f.write(val + '\n')

# Write dense biases (7)
dense_bias_start = 40 + (DENSE_SIZE * DENSE_CLASSES)
with open('modulos_verilog/weights_dense_b.hex', 'w') as f:
    for val in out_lines[dense_bias_start : dense_bias_start + 7]:
        f.write(val + '\n')

# Write dense weights for each class (900 each)
for c in range(DENSE_CLASSES):
    start = 40 + (c * DENSE_SIZE)
    end = start + DENSE_SIZE
    with open(f'modulos_verilog/weights_dense_{c}.hex', 'w') as f:
        for val in out_lines[start:end]:
            f.write(val + '\n')

print("Split weights successfully.")
