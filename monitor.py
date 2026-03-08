import os, time, collections, joblib
import numpy as np
from config import MODEL_PATH, SCAN_INTERVAL_SEC, WINDOW_SIZE, LABEL_NAMES
from scanner import scan_wifi, RSSISimulator
from features import _features_for_signal

def _bar(prob,w=30):
    f=int(round(prob*w)); return chr(9608)*f+chr(9617)*(w-f)

class RealTimeMonitor:
    def __init__(self,use_simulator=True,sim_person=False,smoothing=3):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model not found. Run: python main.py train")
        self.pipe=joblib.load(MODEL_PATH)
        self.sim=RSSISimulator(person_present=sim_person) if use_simulator else None
        self.smoothing=smoothing
        self._buf=collections.deque(maxlen=WINDOW_SIZE)
        self._hist=collections.deque(maxlen=smoothing)
        try: self._fnames=list(self.pipe.named_steps["clf"].feature_names_in_)
        except Exception: self._fnames=[]

    def _scan(self): return self.sim.scan() if self.sim else scan_wifi()

    def _features(self):
        keys=set()
        for s in self._buf: keys.update(k for k in s if k.startswith("rssi_"))
        row={}
        for key in sorted(keys):
            sig=np.array([s.get(key,np.nan) for s in self._buf],dtype=float)
            nm=np.isnan(sig)
            if nm.all(): continue
            sig[nm]=np.nanmean(sig)
            tag=key.replace("rssi_","")
            for fn,val in _features_for_signal(sig).items():
                row[tag+"__"+fn]=val
        stds=[v for k,v in row.items() if k.endswith("__std")]
        rngs=[v for k,v in row.items() if k.endswith("__range")]
        row["agg__mean_std"]=float(np.mean(stds)) if stds else 0.0
        row["agg__max_range"]=float(np.max(rngs)) if rngs else 0.0
        if self._fnames:
            arr=np.array([row.get(f,0.0) for f in self._fnames],dtype=np.float32)
        else:
            arr=np.array(list(row.values()),dtype=np.float32)
        return arr.reshape(1,-1)

    def run(self,duration=60,interval=SCAN_INTERVAL_SEC):
        print("\n"+"="*60)
        print("  WiFi Human Presence Monitor - LIVE  (Ctrl+C to stop)")
        print("="*60+"\n")
        end_t=time.time()+duration if duration>0 else float("inf")
        try:
            while time.time()<end_t:
                t0=time.time(); aps=self._scan()
                s={"timestamp":t0}
                for ap in aps: s["rssi_"+ap.bssid.replace(":","_")]=ap.rssi
                self._buf.append(s)
                if len(self._buf)==WINDOW_SIZE:
                    X=self._features(); pred=int(self.pipe.predict(X)[0])
                    self._hist.append(pred)
                    sm=int(collections.Counter(self._hist).most_common(1)[0][0])
                    try: conf=float(self.pipe.predict_proba(X)[0][sm])
                    except: conf=1.0
                    icon="[NO PERSON]" if sm==0 else "[PERSON!]  "
                    bar=_bar(conf)
                    print("\r  "+icon+"  Conf:["+bar+"] "+str(round(conf*100,1))+"% APs:"+str(len(aps))+"  ",end="",flush=True)
                else:
                    need=WINDOW_SIZE-len(self._buf)
                    print("\r  [Buffering... "+str(need)+" more samples]  ",end="",flush=True)
                sl=interval-(time.time()-t0)
                if sl>0: time.sleep(sl)
        except KeyboardInterrupt:
            print("\n\n[Monitor] Stopped.")
        print()