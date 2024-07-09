import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
EGC24_path = ".../DefiEGC2024/networks/"
work_path = "..."

## The input should be src, dst, total value and number of transactions, and time stamp
egc = pd.read_csv('.../EGC_full_feat'+ '/edges.csv',
                            names=['src', 'dst', 'value', 'nb', 'time'])
egc['time'] = (egc['time'] - 1) / 86400 + 1
egc['time'] = egc['time'].astype(int)
egc.head(3)


### Normalization
egc['label_nb'] = np.log10(egc['nb']+1)
egc['label_nb'] = egc['label_nb']/egc['label_nb'].max()
egc['label_value'] = np.log10(egc['value']+1)
egc['label_value'] = egc['label_value']/(egc['label_value'].max()+1)


### mapping src and dst string to integer
combined_values = pd.concat([egc['src'], egc['dst']]).unique()
combined_index = pd.factorize(combined_values)[0]
mapping_array = np.array([combined_values, combined_index])
mapping_dict = dict(zip(combined_values, combined_index))
egc['src_id'] = egc['src'].map(mapping_dict)
egc['dst_id'] = egc['dst'].map(mapping_dict)

np.save(work_path+"data/egc_node_name_mapping.npy", mapping_array.astype('str'))
egc.to_csv(work_path+"data/processed_edges.csv", index=False)


## Split to subsets
egc = pd.read_csv(work_path+"data/processed_edges.csv", header=0)
egc = egc[["src_id", "dst_id", "time", "label_nb", "label_value"]]


k = 5 # Split to k dataframes having same number of rows
egc = egc.sort_values(by='time')
rows_per_part = len(egc) // k + 1

parts = []
for i in range(0, k):
  parts.append(egc.iloc[i * rows_per_part : min((i+1)*rows_per_part, len(egc))])


### Saving subset
folder_path = work_path+"data/%d_splits_samelen/"%k
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

for i, part_df in enumerate(parts):
    part_file_path = os.path.join(folder_path, "%dsplits_%d.csv"%(k,i))
    part_df.to_csv(part_file_path, index=False)
    print(f"Saved part {k} to {part_file_path}")


