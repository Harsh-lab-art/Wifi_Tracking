import os, glob
import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft
from typing import Tuple, List
from config import WINDOW_SIZE, WINDOW_STEP, DATA_DIR

def _features_for_signal(sig):
    n=len(sig); diffs=np.diff(sig)
    mean=float(np.mean(sig)); std=float(np.std(sig))
    mn=float(np.min(sig)); mx=float(np.max(sig))
    rng=mx-mn; median=float(np.median(sig))
    iqr=float(np.percentile(sig,75)-np.percentile(sig,25))
    skw=float(stats.skew(sig)); kurt=float(stats.kurtosis(sig))
    mad=float(np.mean(np.abs(diffs))) if len(diffs)>0 else 0.0
    zcr=float(np.sum(np.diff(np.sign(sig-mean))!=0))/n
    rms=float(np.sqrt(np.mean(sig**2)))
    spectrum=np.abs(fft(sig-mean))
    dom_pow=float(np.max(spectrum[1:n//2+1])) if n>2 else 0.0
    return dict(mean=mean,std=std,min=mn,max=mx,range=rng,median=median,iqr=iqr,
                skew=skw,kurt=kurt,mad=mad,zcr=zcr,rms=rms,dom_pow=dom_pow)

def load_csv_files(data_dir=DATA_DIR):
    files=glob.glob(os.path.join(data_dir,"*.csv"))
    if not files: raise FileNotFoundError("No CSV files in "+data_dir)
    return pd.concat([pd.read_csv(f) for f in files],ignore_index=True).sort_values("timestamp").reset_index(drop=True)

def extract_features(df, window_size=WINDOW_SIZE, window_step=WINDOW_STEP):
    rssi_cols=[c for c in df.columns if c.startswith("rssi_")]
    if not rssi_cols: raise ValueError("No RSSI columns found.")
    X_rows=[]; y_rows=[]
    for start in range(0,len(df)-window_size+1,window_step):
        window=df.iloc[start:start+window_size]
        label=int(window["label"].value_counts().idxmax())
        row={}
        for col in rssi_cols:
            sig=window[col].dropna().values.astype(float)
            if len(sig)<window_size//2: continue
            if len(sig)<window_size:
                sig=np.pad(sig,(0,window_size-len(sig)),mode="mean")
            tag=col.replace("rssi_","")
            for fn,val in _features_for_signal(sig).items():
                row[tag+"__"+fn]=val
        if not row: continue
        stds=[v for k,v in row.items() if k.endswith("__std")]
        ranges=[v for k,v in row.items() if k.endswith("__range")]
        row["agg__mean_std"]=float(np.mean(stds)) if stds else 0.0
        row["agg__max_range"]=float(np.max(ranges)) if ranges else 0.0
        X_rows.append(row); y_rows.append(label)
    if not X_rows: raise ValueError("No windows extracted.")
    feat_df=pd.DataFrame(X_rows).fillna(0)
    return feat_df.values.astype(np.float32), np.array(y_rows,dtype=np.int32), list(feat_df.columns)