from pathlib import Path
import json, time, math, shutil, zipfile
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

CLASS_NAMES = ['plastic','paper','metal']
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class ManifestDataset(Dataset):
    def __init__(self, resources_dir, manifest_csv, transform=None, dataframe=None):
        self.resources_dir = Path(resources_dir)
        self.df = pd.read_csv(manifest_csv) if dataframe is None else dataframe.reset_index(drop=True).copy()
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = Image.open(self.resources_dir / r['path']).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, int(r['label']), str(r['path'])

class SmallCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Linear(64,num_classes)
    def forward(self,x): return self.classifier(self.features(x).flatten(1))

def seed_everything(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def baseline_transform(size=96, augment=False):
    ops=[transforms.Resize((size,size))]
    if augment:
        ops += [transforms.ColorJitter(brightness=0.35, contrast=0.25)]
    ops += [transforms.ToTensor()]
    return transforms.Compose(ops)

def transfer_transform(size=96, augment=False):
    ops=[transforms.Resize((size,size))]
    if augment:
        ops += [transforms.ColorJitter(brightness=0.25, contrast=0.20)]
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)

def make_loaders(resources_dir, train_manifest, val_manifest, size=96, batch_size=64, augment=False, seed=42, train_df=None, transfer=False):
    tf_train = transfer_transform(size,augment) if transfer else baseline_transform(size,augment)
    tf_val = transfer_transform(size,False) if transfer else baseline_transform(size,False)
    tr=ManifestDataset(resources_dir, train_manifest, tf_train, dataframe=train_df)
    va=ManifestDataset(resources_dir, val_manifest, tf_val)
    g=torch.Generator().manual_seed(seed)
    return (
        DataLoader(tr,batch_size=batch_size,shuffle=True,num_workers=0,generator=g),
        DataLoader(va,batch_size=batch_size,shuffle=False,num_workers=0),
        tr, va
    )

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_all(model): return sum(p.numel() for p in model.parameters())

def train_model(model, train_loader, val_loader, epochs=8, lr=1e-3, weight_decay=0.0, class_weights=None, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model=model.to(device)
    if class_weights is not None:
        class_weights=torch.as_tensor(class_weights,dtype=torch.float32,device=device)
    criterion=nn.CrossEntropyLoss(weight=class_weights)
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=weight_decay)
    history=[]; start=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train(); tr_loss=0.; tr_correct=tr_total=0
        for xb,yb,*_ in train_loader:
            xb,yb=xb.to(device),yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits=model(xb); loss=criterion(logits,yb); loss.backward(); opt.step()
            tr_loss += loss.item()*len(yb); tr_total += len(yb); tr_correct += (logits.argmax(1)==yb).sum().item()
        model.eval(); va_loss=0.; va_correct=va_total=0
        with torch.inference_mode():
            for xb,yb,*_ in val_loader:
                xb,yb=xb.to(device),yb.to(device)
                logits=model(xb); loss=criterion(logits,yb)
                va_loss += loss.item()*len(yb); va_total += len(yb); va_correct += (logits.argmax(1)==yb).sum().item()
        history.append({'epoch':epoch,'train_loss':tr_loss/tr_total,'train_acc':tr_correct/tr_total,'val_loss':va_loss/va_total,'val_acc':va_correct/va_total})
    elapsed=time.perf_counter()-start
    return model, pd.DataFrame(history), elapsed

def predict(model, loader, device=None):
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model=model.to(device); model.eval(); rows=[]
    with torch.inference_mode():
        for xb,yb,paths in loader:
            logits=model(xb.to(device)); probs=logits.softmax(1).cpu(); pred=probs.argmax(1)
            conf=probs.max(1).values
            for p,t,pr,c in zip(paths,yb.tolist(),pred.tolist(),conf.tolist()):
                rows.append({'path':p,'true':int(t),'pred':int(pr),'confidence':float(c),'true_name':CLASS_NAMES[int(t)],'pred_name':CLASS_NAMES[int(pr)]})
    return pd.DataFrame(rows)

def metrics_from_predictions(pred_df):
    y=pred_df.true.to_numpy(); p=pred_df.pred.to_numpy()
    pr,rc,f1,sup=precision_recall_fscore_support(y,p,labels=range(len(CLASS_NAMES)),zero_division=0)
    return {
        'accuracy':float(accuracy_score(y,p)),
        'macro_f1':float(f1_score(y,p,average='macro')),
        'per_class':pd.DataFrame({'class':CLASS_NAMES,'precision':pr,'recall':rc,'f1':f1,'support':sup})
    }

def save_learning_curves(history, path, title='Learning curves'):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(7,4.5))
    ax.plot(history.epoch,history.train_loss,label='train loss'); ax.plot(history.epoch,history.val_loss,label='validation loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title(title); ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)

def save_accuracy_curves(history,path,title='Accuracy curves'):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(7,4.5))
    ax.plot(history.epoch,history.train_acc,label='train accuracy'); ax.plot(history.epoch,history.val_acc,label='validation accuracy')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy'); ax.set_ylim(0,1); ax.set_title(title); ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)

def save_confusion(pred_df,path,title='Confusion matrix'):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    cm=confusion_matrix(pred_df.true,pred_df.pred,labels=range(len(CLASS_NAMES)))
    fig,ax=plt.subplots(figsize=(5.3,5)); ConfusionMatrixDisplay(cm,display_labels=CLASS_NAMES).plot(ax=ax,cmap='Blues',colorbar=False); ax.set_title(title); fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig); return cm

def save_failure_grid(pred_df,resources_dir,path,n=6,title='Incorrect predictions'):
    wrong=pred_df[pred_df.true!=pred_df.pred].copy().sort_values('confidence',ascending=False).head(n)
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    cols=3; rows=max(1,math.ceil(max(1,len(wrong))/cols)); fig,axes=plt.subplots(rows,cols,figsize=(10,3.2*rows)); axes=np.array(axes).reshape(-1)
    for ax in axes: ax.axis('off')
    for ax,(_,r) in zip(axes,wrong.iterrows()):
        img=Image.open(Path(resources_dir)/r.path).convert('RGB'); ax.imshow(img); ax.set_title(f"True: {r.true_name}\nPred: {r.pred_name} ({r.confidence:.2f})",fontsize=9); ax.axis('off')
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)
    return wrong

def plot_sample_grid(resources_dir,manifest_csv,path,n_per_class=4,seed=42,title='Dataset sample grid'):
    df=pd.read_csv(manifest_csv); rng=np.random.default_rng(seed); samples=[]
    for lab in range(3):
        g=df[df.label==lab]; idx=rng.choice(g.index,size=min(n_per_class,len(g)),replace=False); samples.append(g.loc[idx])
    s=pd.concat(samples); fig,axes=plt.subplots(3,n_per_class,figsize=(3*n_per_class,8))
    for ax in np.array(axes).reshape(-1): ax.axis('off')
    for row,lab in enumerate(range(3)):
        g=s[s.label==lab].reset_index(drop=True)
        for col in range(min(n_per_class,len(g))):
            r=g.iloc[col]; axes[row,col].imshow(Image.open(Path(resources_dir)/r.path).convert('RGB')); axes[row,col].set_title(CLASS_NAMES[lab]); axes[row,col].axis('off')
    fig.suptitle(title); fig.tight_layout(); Path(path).parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=150); plt.close(fig)

def compute_class_weights(manifest_csv):
    df=pd.read_csv(manifest_csv); counts=df.label.value_counts().sort_index().reindex(range(3),fill_value=0).to_numpy(dtype=float)
    weights=len(df)/(len(counts)*counts)
    return counts, weights

def load_source_pretrained_resnet(weights_path, num_target_classes=3, freeze_backbone=True):
    ckpt=torch.load(weights_path,map_location='cpu',weights_only=False)
    model=resnet18(weights=None); model.fc=nn.Linear(model.fc.in_features,len(ckpt['source_classes'])); model.load_state_dict(ckpt['state_dict'])
    if freeze_backbone:
        for p in model.parameters(): p.requires_grad=False
    in_features=model.fc.in_features; model.fc=nn.Linear(in_features,num_target_classes)
    return model, ckpt

def candidate_grid(resources_dir,candidates_csv,path,cols=6):
    df=pd.read_csv(candidates_csv); rows=math.ceil(len(df)/cols); fig,axes=plt.subplots(rows,cols,figsize=(2.6*cols,2.7*rows)); axes=np.array(axes).reshape(-1)
    for ax in axes: ax.axis('off')
    for i,(_,r) in enumerate(df.iterrows()):
        axes[i].imshow(Image.open(Path(resources_dir)/r.path).convert('RGB')); axes[i].set_title(Path(r.path).name,fontsize=6); axes[i].axis('off')
    fig.suptitle('Group 5 review candidates'); fig.tight_layout(); Path(path).parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=150); plt.close(fig)

def filter_manifest_by_removed_paths(manifest_csv,removed_paths):
    df=pd.read_csv(manifest_csv); return df[~df.path.isin(set(removed_paths))].reset_index(drop=True)

def prepare_evidence(root,group_number):
    e=Path(root)/'evidence'/f'group_{group_number}'; e.mkdir(parents=True,exist_ok=True); return e

def save_experiment_summary(evidence_dir, rows):
    pd.DataFrame(rows).to_csv(Path(evidence_dir)/'experiment_summary.csv',index=False)

def zip_evidence(evidence_dir,zip_path=None):
    evidence_dir=Path(evidence_dir); zip_path=Path(zip_path or evidence_dir.with_suffix('.zip'))
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in evidence_dir.rglob('*'):
            if p.is_file(): z.write(p,p.relative_to(evidence_dir.parent))
    return zip_path
