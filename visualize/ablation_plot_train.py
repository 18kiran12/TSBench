import numpy as np
import pandas as pd
import pickle
from glob import glob
import os
import pdb
import flatdict
from tqdm import tqdm
import pdb

    
df_hp_dict = {}
root_dirs = [
    # "runs/nlinear_re_oos_bit_ped", 
    # "runs/nlinear_re_oos_elec",
    "/home/kiranmadhusud/research/gluonts_autoforecast/runs/nlinear_re_oos_fredmd",
    # "/home/kiranmadhusud/research/gluonts_autoforecast/runs/nlinear_re_oos_hos_traffic",
    # "/home/kiranmadhusud/research/gluonts_autoforecast/runs/nlinear_re_oos_kdd_covid",
    # "/home/kiranmadhusud/research/gluonts_autoforecast/runs/nlinear_re_oos_m1",
    # "/home/kiranmadhusud/research/gluonts_autoforecast/runs/nlinear_re_oos_m3_month_carparts_m4_daily", # half1
    # "/home/kiranmadhusud/research/gluonts_autoforecast/runs/nlinear_re_oos_m3_y_q",
    # "/home/kiranmadhusud/research/gluonts_autoforecast/runs/nlinear_re_oos_m4",
    # "/home/kiranmadhusud/research/gluonts_autoforecast/runs/nlinear_re_oos_m4mon_m4qaur_temp",
    # "/home/kiranmadhusud/research/gluonts_autoforecast/runs/nlinear_re_oos_nn5",
    # "/home/kiranmadhusud/research/gluonts_autoforecast/runs/nlinear_re_oos_tour",
    # "",
    
]
# log_files = glob(root_dir+'logs/*.log')
for root_dir in tqdm(root_dirs):
    root_dir = root_dir+"/json_metadata/"
    for subdir, dirs, files in os.walk(root_dir):
        # pdb.set_trace()
        print(dirs)
        for file in tqdm(files):
            if file.endswith('.pkl'):
                with open(subdir+file, 'rb') as f:

                    data = pickle.load(f)
                    d = flatdict.FlatDict(data, delimiter =".")
                    data_useful = dict(d)
                    del data_useful["run_folder"]
                    del data_useful["dataset_folder"]
                    # del data_useful["run_folder"]
                    # del 
                    data_useful["val_metrics.val_MASE_monash"]=min(data_useful["val_metrics.MASE_monash"])
                    data_useful["validation_strategy"] = "retrain" if "retrain" in file else "direct"
                    # del data_useful["val_metrics.MASE_monash"]
                    data_useful["model_nlinear.nodes"]=str(data_useful["model_nlinear.nodes"])

                    data_useful["dataset"] = data_useful["dataset"].split(".")[0]
                    if data_useful["dataset"] not in df_hp_dict:
                        df_hp_dict[data_useful["dataset"]] = {}
                    # pdb.set_trace()
                    
                    df_hp_dict[data_useful["dataset"]][data_useful["run_full_name"]] = {k: v for k, v in data_useful.items()}


with open('fred_md_data.pickle', 'wb') as file:
    pickle.dump(df_hp_dict, file)
                    
#     # break
# columns = []
# for col in data_useful.keys():
#     if "." in col:
#         col = col.split(".")[-1]
#     columns.append(col)
    
# df = pd.DataFrame(columns=columns, data=df_list)
# df.to_csv("df_full.csv")



