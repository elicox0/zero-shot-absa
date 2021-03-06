import json
from pathlib import Path
from time import time
from typing import Dict, List

import click
import pandas as pd


def read_json_as_array(json_file: Path, out: str) -> None:
    """
    Read a given Yelp JSON file as string, adding opening / closing
    brackets and commas to convert from separate JSON objects to
    an array of JSON objects, so JSON aware libraries can properly read
    
    Parameters
    -----------
    json_file: path-like
    out      : str
    """

    json_data = ''

    with open(json_file, 'r', encoding='utf-8') as in_file, open(out, 'w') as fout:
        for i, line in enumerate(in_file):
            if i == 0 and line:
                fout.write('[' + line)
            elif line:
                fout.write(',' + line)
            else:
                fout.write(line)
        fout.write(']\n')

def load_json(json_data: str) -> pd.DataFrame:
    """
    Read and normalize a given JSON array into a pandas DataFrame
    Parameters
    -----------
    json_data: str
        String representation of JSON array
    Returns
    -------
    df: pandas.DataFrame
        DataFrame containing the normalized JSON data
    """

    data = json.loads(json_data)
    df = pd.normalize_json(data)

    return df

@click.command()
@click.argument('json-dir', type=click.Path(exists=True, dir_okay=True))
def main(json_dir):
    """
    Read a given directory containing Yelp JSON data and convert those
    files to CSV under 'csv_out' in the same directory
    """
    t0 = time()

    json_dir = Path(json_dir)
    csv_dir = json_dir / "csv_out"
    csv_dir.mkdir(exist_ok=True)

    file_list: List[Path] = json_dir.glob('*.json')

    with click.progressbar(file_list, label='Processing files..') as bar:
        for file in bar:
            temp = str(file).split('.')
            outf = temp[0] + "_formatted" + temp[1]
            csv_file = csv_dir / (file.stem + '.csv')
            data = read_json_as_array(file, outf)
            #df = load_json(data)
            #write_csv(df, csv_file)

    t1 = time()
    mins = (t1 - t0) // 60
    secs = int((t1 - t0) % 60)
    timing = f'Conversion finished in {mins} minutes and {secs} seconds'

    click.secho(timing, fg='green')

if __name__ == '__main__':
    main()
