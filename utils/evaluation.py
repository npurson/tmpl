def accuracy(pred, target):
    return (pred == target).float().mean()
