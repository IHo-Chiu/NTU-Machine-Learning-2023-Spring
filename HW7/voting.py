import csv
import pandas as pd
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--csv', nargs='+', type=str)
parser.add_argument('--out', default='vote.csv')
args = parser.parse_args()

data = {}
for csv_path in args.csv:
    with open(csv_path, newline='') as csvfile:
        rows = csv.reader(csvfile)
        for i, row in enumerate(rows):
            if i == 0:
                continue
            idx, val = row
            if idx not in data:
                data[idx] = []
            data[idx].append(val)
            
result = []
for i in range(len(data)):
    df = pd.DataFrame({'x':data[str(i)]})
    result.append(df['x'].value_counts().index.tolist()[0])
    print(f'voting ... {i*100//len(data)}%', end ='\r')
print(f'voting ... done')
    

with open(args.out, 'w') as csvfile:
    csvfile.write('ID,Answer\n')
    for i, y in enumerate(result):
        csvfile.write('{},{}\n'.format(i, y))
        
        print(f'saving ... {i*100//len(result)}%', end ='\r')
    print(f'saving ... done')