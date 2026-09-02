import os
import struct
from pathlib import Path
from typing import List, Tuple, Union, Any

def write_bmp(file_path: Union[str, Path], grid: List[List[Tuple[int, int, int]]]) -> None:
    height = len(grid)
    width = len(grid[0])
    row_padding = (4 - (width * 3) % 4) % 4
    image_size = (width * 3 + row_padding) * height
    file_size = 54 + image_size

    bmp_header = struct.pack("<2sIHHI", b"BM", file_size, 0, 0, 54)
    dib_header = struct.pack("<IIIHHIIIIII", 40, width, height, 1, 24, 0, image_size, 2835, 2835, 0, 0)

    with open(file_path, "wb") as f:
        f.write(bmp_header)
        f.write(dib_header)
        for y in range(height - 1, -1, -1):
            row_bytes = bytearray()
            for x in range(width):
                r, g, b = grid[y][x]
                row_bytes.extend([b & 0xFF, g & 0xFF, r & 0xFF])
            row_bytes.extend([0] * row_padding)
            f.write(row_bytes)

def read_bmp(file_path: Union[str, Path]) -> List[List[Tuple[int, int, int]]]:
    with open(file_path, "rb") as f:
        data = f.read()

    if len(data) < 54 or data[:2] != b"BM":
        raise ValueError("Invalid BMP file header")

    offset = struct.unpack("<I", data[10:14])[0]
    width, height, planes, bpp = struct.unpack("<IIHH", data[18:30])
    
    if bpp != 24:
        raise ValueError(f"Expected 24 bpp, got {bpp}")

    row_padding = (4 - (width * 3) % 4) % 4
    grid = [[(0, 0, 0) for _ in range(width)] for _ in range(height)]
    
    pos = offset
    for y in range(height - 1, -1, -1):
        for x in range(width):
            b = data[pos]
            g = data[pos + 1]
            r = data[pos + 2]
            grid[y][x] = (r, g, b)
            pos += 3
        pos += row_padding

    return grid

def read_image_pixels(file_path: Union[str, Path]) -> List[List[Tuple[int, int, int]]]:
    path_str = str(file_path)
    try:
        from PIL import Image
        img = Image.open(path_str).convert("RGB")
        w, h = img.size
        grid = []
        for y in range(h):
            row = [img.getpixel((x, y)) for x in range(w)]
            grid.append(row)
        return grid
    except Exception:
        if path_str.lower().endswith(".bmp"):
            return read_bmp(path_str)
        raise ValueError(f"Cannot read image file {path_str}")
