import os, joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.pipeline import Pipeline
from config import MODEL_PATH, MODEL_DIR, LOG_DIR, RANDOM_STATE, TEST_SIZE, CV_FOLDS, LABEL_NAMES

def _build(mtype="rf"):
    if mtype=="rf":   clf=RandomForestClassifier(n_estimators=200,min_samples_leaf=2,n_jobs=-1,random_state=RANDOM_STATE)
    elif mtype=="gb": clf=GradientBoostingClassifier(n_estimators=150,learning_rate=0.1,max_depth=4,random_state=RANDOM_STATE)
    elif mtype=="svm":clf=SVC(kernel="rbf",C=10,gamma="scale",probability=True,random_state=RANDOM_STATE)
    else: raise ValueError("Unknown: "+mtype)
    return Pipeline([("scaler",StandardScaler()),("clf",clf)])

def train(X, y, feature_names, model_type="rf", plot=True):
    os.makedirs(MODEL_DIR,exist_ok=True); os.makedirs(LOG_DIR,exist_ok=True)
    print("\n[Trainer] Dataset: "+str(X.shape[0])+" windows x "+str(X.shape[1])+" features")
    uniq,cnts=np.unique(y,return_counts=True)
    print("[Trainer] Labels: "+str(dict(zip(uniq.tolist(),cnts.tolist()))))
    X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=TEST_SIZE,random_state=RANDOM_STATE,stratify=y)
    pipe=_build(model_type)
    min_class=min(np.bincount(y_tr))
    n_splits=min(CV_FOLDS,int(min_class))
    if n_splits<2: n_splits=2
    cv=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=RANDOM_STATE)
    cv_sc=cross_val_score(pipe,X_tr,y_tr,cv=cv,scoring="accuracy",n_jobs=-1)
    print("[Trainer] "+str(n_splits)+"-fold CV: "+str(round(cv_sc.mean(),3))+" +/- "+str(round(cv_sc.std(),3)))
    pipe.fit(X_tr,y_tr); y_pred=pipe.predict(X_te)
    print("[Trainer] Test acc: "+str(round(accuracy_score(y_te,y_pred),3))+"\n")
    print(classification_report(y_te,y_pred,target_names=[LABEL_NAMES[0],LABEL_NAMES[1]]))
    joblib.dump(pipe,MODEL_PATH)
    print("[Trainer] Model saved -> "+MODEL_PATH)
    if plot:
        cm=confusion_matrix(y_te,y_pred)
        fig,ax=plt.subplots(figsize=(5,4))
        sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax,
                    xticklabels=[LABEL_NAMES[0],LABEL_NAMES[1]],yticklabels=[LABEL_NAMES[0],LABEL_NAMES[1]])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
        plt.tight_layout(); fig.savefig(os.path.join(LOG_DIR,"confusion_matrix.png"),dpi=150); plt.close(fig)
        clf=pipe.named_steps["clf"]
        if hasattr(clf,"feature_importances_"):
            imp=clf.feature_importances_; top=min(20,len(imp)); idx=np.argsort(imp)[-top:][::-1]
            fig,ax=plt.subplots(figsize=(8,5))
            ax.barh([feature_names[i] for i in idx[::-1]],imp[idx[::-1]],color="#2196F3")
            ax.set_xlabel("Importance"); ax.set_title("Top Feature Importances")
            plt.tight_layout(); fig.savefig(os.path.join(LOG_DIR,"feature_importance.png"),dpi=150); plt.close(fig)
        try:
            proba=pipe.predict_proba(X_te)[:,1]
            fpr,tpr,_=roc_curve(y_te,proba); rauc=auc(fpr,tpr)
            fig,ax=plt.subplots(figsize=(5,4))
            ax.plot(fpr,tpr,lw=2,label="AUC="+str(round(rauc,3))); ax.plot([0,1],[0,1],"k--",lw=1)
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve"); ax.legend()
            plt.tight_layout(); fig.savefig(os.path.join(LOG_DIR,"roc_curve.png"),dpi=150); plt.close(fig)
            print("[Trainer] Plots saved -> "+LOG_DIR)
        except Exception: pass
    return pipe