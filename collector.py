import os, csv, time, datetime
from config import (SCAN_INTERVAL_SEC, COLLECTION_DURATION, DATA_DIR,
                    LABEL_NO_PERSON, LABEL_PERSON, LABEL_NAMES)
from scanner import scan_wifi, RSSISimulator

def _ts():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def collect_session(label, duration=COLLECTION_DURATION, interval=SCAN_INTERVAL_SEC,
                    use_simulator=True, verbose=True):
    os.makedirs(DATA_DIR, exist_ok=True)
    lstr = LABEL_NAMES[label].replace(" ","_")
    fname = os.path.join(DATA_DIR, lstr+"_"+_ts()+".csv")
    sim = RSSISimulator(person_present=(label==LABEL_PERSON)) if use_simulator else None
    if verbose:
        print("\n[Collector] Recording " + LABEL_NAMES[label] + " for " + str(duration) + "s")
    rows = []; end_t = time.time()+duration; step = 0
    while time.time() < end_t:
        t0 = time.time()
        aps = sim.scan() if sim else scan_wifi()
        row = {"timestamp": time.time(), "label": label}
        for ap in aps:
            row["rssi_"+ap.bssid.replace(":","_")] = ap.rssi
        rows.append(row); step += 1
        if verbose and step % 10 == 0:
            elapsed = duration-(end_t-time.time())
            print("  "+str(round(elapsed))+"/"+str(duration)+"s — APs:"+str(len(aps))+
                  " RSSI:"+str([a.rssi for a in aps[:3]]))
        sl = interval-(time.time()-t0)
        if sl > 0: time.sleep(sl)
    if not rows: raise RuntimeError("No data.")
    fieldnames = list(rows[0].keys())
    for r in rows:
        for f in fieldnames: r.setdefault(f,None)
    with open(fname,"w",newline="") as fh:
        w = csv.DictWriter(fh,fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    if verbose: print("[Collector] Saved "+str(len(rows))+" samples -> "+fname+"\n")
    return fname

def collect_dataset(sessions_per_class=3, duration=COLLECTION_DURATION, use_simulator=True):
    files = []
    for label in [LABEL_NO_PERSON, LABEL_PERSON]:
        for i in range(sessions_per_class):
            print("\n=== Session "+str(i+1)+"/"+str(sessions_per_class)+" - "+LABEL_NAMES[label]+" ===")
            if not use_simulator: input("  Press ENTER when ready...")
            files.append(collect_session(label,duration=duration,use_simulator=use_simulator))
    return files