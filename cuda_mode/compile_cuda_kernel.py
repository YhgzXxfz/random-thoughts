"""
Example:
$ python -m compile_cuda_kernel \
    --name=matmul \
    --cu_file=matrix_multiplication.cu \
    --cpp_source="torch::Tensor matmul(torch::Tensor m, torch::Tensor n);" \
    --funcs=matmul
"""

import argparse
from pathlib import Path

from torch.utils.cpp_extension import load_inline


def compile_extension(name: str, cu_file: str, cpp_source: str, funcs: str):
    cuda_source = Path(cu_file).read_text()

    # Load the CUDA kernel as a PyTorch extension
    module = load_inline(
        name=name,
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=funcs.split(","),
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        # build_directory='./cuda_build',
    )
    return module


def main():
    parser = argparse.ArgumentParser(description="Compile CUDA Kernel.")
    parser.add_argument("--name", type=str, help="The name of output kernel.")
    parser.add_argument("--cu_file", type=str, help="Path to kernel file.")
    parser.add_argument("--cpp_source", type=str, help="cpp source.")
    parser.add_argument("--funcs", type=str, help="Function names concatenated with ,")
    args = parser.parse_args()
    compile_extension(args.name, args.cu_file, args.cpp_source, args.funcs)


if __name__ == "__main__":
    main()
