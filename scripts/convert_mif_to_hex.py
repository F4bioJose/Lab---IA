import re

with open('modulos_verilog/weights_all.mif', 'r') as f:
    lines = f.readlines()

out_lines = []
for line in lines:
    m = re.match(r'^\s*\d+\s*:\s*([0-9a-fA-F]+)\s*;', line)
    if m:
        out_lines.append(m.group(1))

with open('modulos_verilog/weights_all.hex', 'w') as f:
    for val in out_lines:
        f.write(val + '\n')

print(f"Converted {len(out_lines)} words.")
