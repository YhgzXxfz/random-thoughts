from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline
from torchvision.io import read_image, write_png


def compile_extension():
    cuda_source = Path("1d_rgb_to_grayscale_kernel.cu").read_text()
    cpp_source = "torch::Tensor rgb_to_grayscale(torch::Tensor image);"

    # Load the CUDA kernel as a PyTorch extension
    rgb_to_grayscale_extension = load_inline(
        name="1d_rgb_to_grayscale_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["rgb_to_grayscale"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        # build_directory='./cuda_build',
    )
    return rgb_to_grayscale_extension


def main():
    """
    Use torch cpp inline extension function to compile the kernel in grayscale_kernel.cu.
    Read input image, convert it to grayscale via custom cuda kernel and write it out as png.
    """
    ext = compile_extension()

    x = read_image("puppy.jpg").contiguous().cuda()
    print("mean:", x.float().mean())
    print("Input image:", x.shape, x.dtype)

    assert x.dtype == torch.uint8

    y = ext.rgb_to_grayscale(x)

    print("Output image:", y.shape, y.dtype)
    print("mean", y.float().mean())
    write_png(
        y.cpu().unsqueeze(-1).permute(2, 0, 1), "my_output.png"
    )  # Requires 3 dims and channels must in the 0 dimension.


if __name__ == "__main__":
    main()
