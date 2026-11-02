"""Shape and gradient smoke test for the IR&ArF adapter."""
from types import SimpleNamespace
import torch
from model import Main

def main():
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args=SimpleNamespace(bands=103,msi_channels=4,gpu=device.index or 0)
    model=Main(bands=args.bands,args=args).to(device)
    lr=torch.rand(1,103,16,16,device=device)
    msi=torch.rand(1,4,64,64,device=device)
    outputs=model(msi,lr,64)
    pred=outputs[0]
    assert pred.shape==(1,103,64,64),tuple(pred.shape)
    assert torch.isfinite(pred).all()
    pred.abs().mean().backward()
    grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    print("IR&ArF smoke test passed")
    print("device:",device,"output:",tuple(pred.shape))
    print("parameters:",sum(p.numel() for p in model.parameters()))
if __name__=="__main__":
    main()
