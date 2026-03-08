import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def cmd_collect(args):
    from collector import collect_dataset
    collect_dataset(sessions_per_class=args.sessions,duration=args.duration,use_simulator=not args.live)

def cmd_train(args):
    from features import load_csv_files, extract_features
    from trainer import train
    print("[Main] Loading data...")
    df=load_csv_files()
    print("[Main] "+str(len(df))+" samples. Extracting features...")
    X,y,fn=extract_features(df)
    print("[Main] "+str(X.shape[0])+" windows x "+str(X.shape[1])+" features")
    train(X,y,fn,model_type=args.model,plot=not args.no_plot)

def cmd_monitor(args):
    from monitor import RealTimeMonitor
    RealTimeMonitor(use_simulator=not args.live,sim_person=args.person,smoothing=3).run(duration=args.duration)

def cmd_demo(args):
    print("\n"+"="*60)
    print("  WiFi Human Presence Detection - Full Demo")
    print("="*60)
    print("\n--- Step 1/3: Data Collection ---")
    from collector import collect_dataset
    collect_dataset(sessions_per_class=3,duration=20,use_simulator=True)
    print("\n--- Step 2/3: Model Training ---")
    from features import load_csv_files, extract_features
    from trainer import train
    df=load_csv_files(); X,y,fn=extract_features(df)
    train(X,y,fn,model_type="rf",plot=True)
    print("\n--- Step 3/3: Real-Time Monitoring ---")
    from monitor import RealTimeMonitor
    RealTimeMonitor(use_simulator=True,sim_person=True,smoothing=3).run(duration=args.duration)
    print("\nDone! Check logs/ folder for plots.")

def main():
    p=argparse.ArgumentParser(description="WiFi Human Presence Detection")
    sub=p.add_subparsers(dest="command")
    c=sub.add_parser("collect"); c.add_argument("--sessions",type=int,default=3); c.add_argument("--duration",type=int,default=30); c.add_argument("--live",action="store_true")
    t=sub.add_parser("train");   t.add_argument("--model",choices=["rf","gb","svm"],default="rf"); t.add_argument("--no-plot",action="store_true",dest="no_plot")
    m=sub.add_parser("monitor"); m.add_argument("--duration",type=int,default=60); m.add_argument("--live",action="store_true"); m.add_argument("--person",action="store_true")
    d=sub.add_parser("demo");    d.add_argument("--duration",type=int,default=20)
    args=p.parse_args()
    if not args.command: p.print_help(); return
    for attr,default in [("sessions",3),("duration",30),("live",False),("model","rf"),("no_plot",False),("person",False)]:
        if not hasattr(args,attr): setattr(args,attr,default)
    {"collect":cmd_collect,"train":cmd_train,"monitor":cmd_monitor,"demo":cmd_demo}[args.command](args)

if __name__=="__main__":
    main()