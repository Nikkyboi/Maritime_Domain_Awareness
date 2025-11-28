predictedPoint = torch.cat(predictedPoint, dim=0)
        actualPoint = torch.cat(actualPoint, dim=0)
        predictedPoint = predictedPoint.squeeze(1)
        actualPoint = actualPoint.squeeze(1)