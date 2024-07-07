import numpy as np
import torch


def predict(
    model, dataloader, device
    ):
    
    torch.cuda.empty_cache()
    model.eval()
    output = []

    with torch.no_grad():
        for _, (x, _) in enumerate(dataloader):
            x = x.to(device=device)

            y_pred = model(x)
            y_pred = y_pred.softmax(-3).argmax(-3)

            output.extend(y_pred.cpu().numpy())
    torch.cuda.empty_cache()

    return np.array(output)