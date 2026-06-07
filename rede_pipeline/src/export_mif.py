import numpy as np
import tensorflow as tf
from src.config import Config
from src.model_io import load_tinycnn_model

cfg = Config()

class MIFExporter:
    def __init__(self, bit_width=8, frac_bits=7):
        self.bit_width = bit_width
        self.frac_bits = frac_bits
        self.max_val = (2 ** (bit_width - 1)) - 1
        self.min_val = -(2 ** (bit_width - 1))

    def to_fixed_point(self, value):
        fixed_val = int(np.round(value * (2 ** self.frac_bits)))
        fixed_val = max(min(fixed_val, self.max_val), self.min_val)
        if fixed_val < 0:
            fixed_val = (1 << self.bit_width) + fixed_val
        return format(fixed_val, f'0{self.bit_width // 4}X')

    def generate_single_mif(self, model, filename="all_weights"):
        conv_layers = [l for l in model.layers if type(l).__name__ == "Conv2D"]
        dense_layers = [l for l in model.layers if type(l).__name__ == "Dense"]
        
        if not conv_layers or not dense_layers:
            raise RuntimeError("Modelo sem Conv2D ou Dense")

        conv_w, conv_b = conv_layers[0].get_weights()
        dense_w, dense_b = dense_layers[-1].get_weights()

        all_weights = []

        # 1. Conv weights: shape (3,3,1,4) -> filter by filter
        n_filters = conv_w.shape[3]
        for f in range(n_filters):
            kernel = conv_w[:, :, 0, f].flatten()
            all_weights.extend(kernel.tolist())
        
        # 2. Conv biases
        all_weights.extend(conv_b.tolist())

        # 3. Dense weights: shape (900, 19) -> column major
        for cls in range(dense_w.shape[1]):
            all_weights.extend(dense_w[:, cls].tolist())
            
        # 4. Dense biases
        all_weights.extend(dense_b.tolist())

        flat_data = np.array(all_weights)
        estimated_bytes = len(flat_data) * (self.bit_width // 8)
        if estimated_bytes > cfg.MAX_MIF_BYTES:
            raise RuntimeError(f"Pesos excedem limite da FPGA: {estimated_bytes} bytes > {cfg.MAX_MIF_BYTES} bytes.")

        mif_content = [
            f"DEPTH = {len(flat_data)};",
            f"WIDTH = {self.bit_width};",
            "ADDRESS_RADIX = HEX;",
            "DATA_RADIX = HEX;",
            "CONTENT BEGIN",
        ]
        hex_content = []
        for addr, val in enumerate(flat_data):
            fixed_hex = self.to_fixed_point(val)
            mif_content.append(f"{addr:X} : {fixed_hex};")
            hex_content.append(fixed_hex)
        mif_content.append("END;")

        # Exporta MIF
        mif_path = cfg.EXPORT_DIR / f"{filename}.mif"
        mif_path.parent.mkdir(exist_ok=True, parents=True)
        with open(mif_path, "w") as f:
            f.write("\n".join(mif_content))
            
        # Exporta HEX
        hex_path = cfg.EXPORT_DIR / f"{filename}.hex"
        with open(hex_path, "w") as f:
            f.write("\n".join(hex_content))
            
        print(f" -> Exportado: {mif_path.name} e {hex_path.name} ({len(flat_data)} words)")

def export_model_to_mif():
    model_path = cfg.MODEL_PATH
    if not model_path.exists(): return
    model = load_tinycnn_model(model_path, compile_model=False)
    exporter = MIFExporter()
    print("\n[MIF] Iniciando exportação dos pesos quantizados num único arquivo...")
    exporter.generate_single_mif(model, "all_weights")
