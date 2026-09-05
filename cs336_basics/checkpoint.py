import torch

def save_checkpoint(model, optimizer, iteration, out):
    # step 1: 把 model / optimizer 的 state_dict 和 iteration 打包成一个 dict
    obj = {
        "model":model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    
    # step 2: torch.save(obj, out)
    torch.save(obj, out)


def load_checkpoint(src, model, optimizer):
    # step 1: torch.load 读回字典
    obj = torch.load(src)

    # step 2: 分别把状态灌回传入的 model / optimizer（load_state_dict）
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])

    # step 3: 返回存档时的 iteration
    return obj["iteration"]