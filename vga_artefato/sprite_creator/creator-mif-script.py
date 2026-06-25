import cv2

def png_to_mif(image_path, mif_path, width = 256, height = 608):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # binarização - mais claro que 127, vira 1 (branco) e o restante é 0
    _, img_binaria = cv2.threshold(image ,127,1,cv2.THRESH_BINARY)

    N = width * height

    # cabeçalho da Altera

    header = (
        f"DEPTH= {N};\n"
        f"WIDTH = 1;\n" # cada bit representará um pixel do sprite
        f"ADDRESS_RADIX = DEC;\n" # endereco
        f"DATA_RADIX = BIN;\n\n"
        f"CONTENT BEGIN\n"
    )

    with open(mif_path, 'w') as f:
        f.write(header)

        endereco = 0
        # a partir da primeira coluna, varrer a primeira linha completamente e assim sucessivamente - cima para baixo, esquerda para direita 
        for y in range(height):
            for x in range(width):
                pixel = img_binaria[y, x]

                f.write(f"{endereco} : {pixel};\n")
                endereco += 1

        f.write("END;\n")

png_to_mif('sprite-alunos.png', 'rom_sprites.mif')