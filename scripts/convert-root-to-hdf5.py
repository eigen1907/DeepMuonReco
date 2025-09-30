#!/usr/bin/env python
import argparse
import uproot
from pathlib import Path
import h5py as h5
import numpy as np


def convert(input_file_path: Path, treepath: str):
    if not input_file_path.exists():
        raise FileNotFoundError(f'Input file does not exist: {input_file_path}')
    print(f'Converting {input_file_path} to HDF5 format...')

    output_file_path = input_file_path.with_suffix('.h5')
    if output_file_path.exists():
        raise FileExistsError(f'Output file already exists: {output_file_path}')
    print(f'Output file will be: {output_file_path}')

    print(f'Reading data from {input_file_path}:{treepath}...')
    with uproot.open(input_file_path) as input_file:
        data = input_file[treepath].arrays(library='np')

    with h5.File(output_file_path, 'w') as output_file:
        for key, value in data.items():
            print(f'Processing {key}')
            if value[0].dtype == np.float64:
                value = np.vectorize(lambda each: each.astype(np.float32), otypes=[object])(value)
            output_file.create_dataset(
                name=key,
                shape=(len(value), ),
                dtype=h5.vlen_dtype(value[0].dtype),
                data=value,
            )



def run(input_file_path_list: list[Path], treepath: str):

    for input_file_path in input_file_path_list:
        convert(input_file_path, treepath)


def main():
    parser = argparse.ArgumentParser(description='Convert ROOT file to HDF5 format.')
    parser.add_argument('-i', '--input', dest='input_file_path_list', type=Path, nargs='+', help='Path to the input ROOT file.')
    parser.add_argument('--treepath', type=str, default='muons1stStep/event', help='Path to the tree inside the ROOT file.')
    args = parser.parse_args()

    run(**vars(args))


if __name__ == '__main__':
    main()
