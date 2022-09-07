import os
from budespack.data.writer import PulWork
from multiprocessing import Pool
import pandas as pd
import glob

if __name__ == '__main__':
    data = 'data'
    trains = pd.read_csv(f'{data}/train.csv')

    els = trains['event'].unique()
    matches = dict()
    for key,el in enumerate(els):
        matches[el] = key
   
    os.makedirs(f'{data}/wds', exist_ok=True)
    puw = PulWork(trains, matches,f'{data}/wds/totar2')
    files_to_record = glob.glob(f"{data}/train/*")
    with Pool(5) as p:
            print(p.map(puw.reocrd_entry, files_to_record))
