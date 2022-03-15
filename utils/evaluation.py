def accuracy(pred, target):
    """
    Args:
        pred: Tensor of size `B x C`
        target: Tensor of size `B`
    """
    return (pred.argmax(dim=1) == target).float().mean()
