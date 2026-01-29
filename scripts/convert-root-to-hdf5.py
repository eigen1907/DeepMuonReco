#!/usr/bin/env python
import argparse
import uproot
from pathlib import Path
import h5py as h5
import numpy as np


def convert(input_file_path: Path, treepath: str, step_size: str = "4 GB"):
    if not input_file_path.exists():
        raise FileNotFoundError(f'Input file does not exist: {input_file_path}')
    print(f'Converting {input_file_path} to HDF5 format (Batch mode)...')

    output_file_path = input_file_path.with_suffix('.h5')
    if output_file_path.exists():
        raise FileExistsError(f'Output file already exists: {output_file_path}')
    print(f'Output file will be: {output_file_path}')

    with h5.File(output_file_path, 'w') as output_file:
        
        iterator = uproot.iterate(
            f"{input_file_path}:{treepath}",
            step_size=step_size,
            library='np'
        )

        for i, batch in enumerate(iterator):
            print(f'Processing batch {i+1}...')
            
            for key, value in batch.items():
                if len(value) > 0 and isinstance(value[0], np.ndarray) and value[0].dtype == np.float64:
                    value = np.vectorize(lambda each: each.astype(np.float32), otypes=[object])(value)
                
                if key not in output_file:
                    dt = h5.vlen_dtype(value[0].dtype) if value.dtype == 'O' else value.dtype
                    
                    output_file.create_dataset(
                        name=key,
                        data=value,
                        maxshape=(None,),
                        chunks=True,
                        dtype=dt
                    )
                
                else:
                    dset = output_file[key]
                    dset.resize((dset.shape[0] + value.shape[0]), axis=0)
                    dset[-value.shape[0]:] = value

    print("Conversion complete.")

def run(input_file_path_list: list[Path], treepath: str, step_size: str):
    for input_file_path in input_file_path_list:
        convert(input_file_path, treepath, step_size)


def main():
    parser = argparse.ArgumentParser(description='Convert ROOT file to HDF5 format.')
    parser.add_argument('-i', '--input', dest='input_file_path_list', type=Path, nargs='+', help='Path to the input ROOT file.')
    parser.add_argument('--treepath', type=str, default='deepMuonRecoNtuplizer/tree', help='Path to the tree inside the ROOT file.')
    parser.add_argument('--step-size', type=str, default='4 GB', help='Batch size for processing.') 
    args = parser.parse_args()

    run(**vars(args))


if __name__ == '__main__':
    main()
