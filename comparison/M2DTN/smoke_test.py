"""Forward/backward smoke test for the M2DTN benchmark adapter."""
import torch
from model import build_m2dtn, architecture_metadata

def main():
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=build_m2dtn(7,3,4).to(device)
    lr=torch.rand(1,7,16,16,device=device)
    msi=torch.rand(1,3,64,64,device=device)
    target=torch.rand(1,7,64,64,device=device)
    pred=model(lr,msi)
    assert pred.shape==target.shape,(pred.shape,target.shape)
    loss=torch.nn.functional.l1_loss(pred,target)
    loss.backward()
    assert bool(torch.isfinite(loss))
    print("M2DTN smoke test passed",device,architecture_metadata(model),"loss",float(loss))

if __name__=="__main__":
    main()
