import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, Any

SENSOR_COLS = ['Amplitude - 402b', 'Amplitude - 602b', 'Amplitude - 502b', 
               'Amplitude - 702b', 'Amplitude - tgs8100', 'Amplitude - 802b']

def load_custom_dataset(data_dir: str = r"C:\Users\HARSH\Downloads\Telegram Desktop",
                        window_size: int = 1024) -> Dict[str, Any]:
    """
    Loads all custom e-nose excel files, extracts labels from sheet names,
    adds metadata columns, segments data into windows and returns raw/windowed data.
    """
    files = ["20-01-2020.xlsx", "20-12-2020_1.xlsx", "21-10-22.xlsx", "21-12-21.xlsx"]
    
    raw_data_list = []
    windowed_data = []
    
    for f in tqdm(files, desc="Loading Excel files"):
        file_path = os.path.join(data_dir, f)
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found.")
            continue
            
        try:
            xl = pd.ExcelFile(file_path)
            for sheet_name in xl.sheet_names:
                df = xl.parse(sheet_name)
                
                # Parse sheet name
                parts = sheet_name.split(',')
                labels = {}
                for p in parts:
                    if '=' in p:
                        k, v = p.split('=')
                        labels[k.strip()] = 1 if int(v) == 1 else 0
                
                h = labels.get('h', 0)
                n = labels.get('n', 0)
                c = labels.get('c', 0)
                e = labels.get('e', 0)
                
                df['h'] = h
                df['n'] = n
                df['c'] = c
                df['e'] = e
                df['date'] = f.split('.')[0]
                df['gas_combination'] = sheet_name
                
                raw_data_list.append(df)
                
                # Segment into windows
                n_rows = len(df)
                for i in range(0, n_rows, window_size):
                    window = df.iloc[i:i+window_size].copy()
                    if len(window) == window_size:
                        windowed_data.append({
                            'window': window[SENSOR_COLS].values,
                            'labels': {'h': h, 'n': n, 'c': c, 'e': e},
                            'date': df['date'].iloc[0],
                            'gas_combination': sheet_name
                        })
        except Exception as e:
            print(f"Error processing {f}: {e}")
                    
    if not raw_data_list:
        return {'raw_data': pd.DataFrame(), 'windowed_data': []}
        
    full_raw_data = pd.concat(raw_data_list, ignore_index=True)
    return {'raw_data': full_raw_data, 'windowed_data': windowed_data}
