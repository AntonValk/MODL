from torchmetrics import Metric
from torchmetrics.aggregation import RunningMean
import torch


class CumulativeError(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("accumulated", default=torch.FloatTensor([0.0]), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape[0] == target.shape[0]
        self.accumulated += int(torch.argmax(preds, axis=1) != target) 

    def compute(self):
        return self.accumulated

class MovingWindowAccuracy(Metric):
    def __init__(self, dist_sync_on_step=False, window_size=50):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.metric = RunningMean(window=window_size)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.metric(int(torch.argmax(preds, axis=1) == target))
        
    def compute(self):
        return self.metric.compute()
        
class NormalizedCumulativeError(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("accumulated", default=torch.FloatTensor([0.0]), dist_reduce_fx="sum")
        self.add_state("count", default=torch.LongTensor([0.0]), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.view(-1, 2)
        target = target.view(-1)

        assert preds.shape[0] == target.shape[0]

        self.accumulated += int(torch.argmax(preds, axis=1) != target) 
        self.count += target.shape[0]
        # print(self.accumulated, self.count)

    def compute(self):
        return self.accumulated / (self.count)

    
class SmoothedCumulativeError(Metric):
    def __init__(self, dist_sync_on_step=False, alpha=0.1):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("accumulated", default=torch.FloatTensor([0.0]), dist_reduce_fx="sum")
        self.alpha = alpha

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.view(-1, 2)
        target = target.view(-1)

        assert preds.shape[0] == target.shape[0]

        self.accumulated = (1-self.alpha) * self.accumulated + (self.alpha) * int(torch.argmax(preds, axis=1) != target) 

    def compute(self):
        return self.accumulated
