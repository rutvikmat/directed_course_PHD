"""Optional PyTorch LSTM baseline. Install torch to enable."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from directed_course.Module_2.M2.Dynamical_Structure_Research.src.pipeline import metrics

def run(frame, tr, va, te, epochs=12, seed=42):
    try:
        import torch
        from torch import nn
    except ImportError:
        print('PyTorch not installed; skipping LSTM. Install: pip install torch')
        return None
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    # Each example is 24 historical observations ending at feature timestamp t.
    lagcols=[f'lag_{i}' for i in range(23,-1,-1)]
    missing=[c for c in lagcols if c not in frame.columns]
    if missing:
        raise ValueError('LSTM requires contiguous lag_0 through lag_23 features')
    X=frame[lagcols].to_numpy(dtype=np.float32)
    y=frame.target.to_numpy(dtype=np.float32)
    scaler=StandardScaler().fit(X[tr].reshape(-1,1))
    X=scaler.transform(X.reshape(-1,1)).reshape(-1,24,1).astype(np.float32)
    ys=StandardScaler().fit(y[tr].reshape(-1,1))
    yscaled=ys.transform(y.reshape(-1,1)).astype(np.float32)
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm=nn.LSTM(1,32,batch_first=True)
            self.head=nn.Linear(32,1)
        def forward(self,x):
            h,_=self.lstm(x)
            return self.head(h[:,-1,:])
    net=Net()
    opt=torch.optim.Adam(net.parameters(),lr=.002)
    loss_fn=nn.MSELoss()
    best=float('inf'); best_state=None; patience=0
    for epoch in range(epochs):
        net.train()
        order=torch.randperm(len(tr))
        for batch in order.split(256):
            ids=tr[batch.numpy()]
            xb=torch.from_numpy(X[ids]); yb=torch.from_numpy(yscaled[ids])
            opt.zero_grad(); loss=loss_fn(net(xb),yb); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vloss=float(loss_fn(net(torch.from_numpy(X[va])),torch.from_numpy(yscaled[va])))
        print(f'LSTM epoch {epoch+1}: validation MSE={vloss:.5f}')
        if vloss < best-1e-5:
            best=vloss; best_state={k:v.detach().clone() for k,v in net.state_dict().items()}; patience=0
        else:
            patience+=1
            if patience>=4: break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred=net(torch.from_numpy(X[te])).numpy()
    pred=ys.inverse_transform(pred).ravel()
    return pred, metrics(y[te],pred)
