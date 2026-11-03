"""Sample-weighted interval diagnostics, summed across every participating rank."""
import torch


class LossMetrics:
    # Row 0: weighted sum, unweighted sum, count. Rows 1..10: same per t bin.
    def __init__(self, device, num_bins=10):
        self.num_bins = num_bins
        self.sums = torch.zeros(num_bins + 1, 3, device=device, dtype=torch.float64)

    def add(self, loss, details=None):
        value = loss.detach().double()
        mse = value if details is None else details["unweighted_mse"].detach().double()
        row = torch.stack((value.reshape(()), mse.reshape(()), value.new_ones(())))
        self.sums[0].add_(row)
        if details is not None:
            # Bins use the actual (possibly BF16-quantized) scheduler timestep,
            # normalized to [0,1]. The final bin includes the endpoint t=1.
            index = (details["normalized_timestep"].detach().double() * self.num_bins)
            index = index.floor().long().clamp(0, self.num_bins - 1).reshape(1) + 1
            self.sums.index_add_(0, index, row.reshape(1, 3))

    def reset(self):
        self.sums.zero_()

    @staticmethod
    def summarize(sums):
        # Called only at a logging/eval boundary after the distributed SUM.
        values = sums.detach().cpu().tolist()
        result = dict(weighted_loss=values[0][0] / max(values[0][2], 1),
                      unweighted_mse=values[0][1] / max(values[0][2], 1),
                      sample_count=int(values[0][2]), bins={})
        n = len(values) - 1
        for i, (weighted, unweighted, count) in enumerate(values[1:]):
            if count:
                result["bins"][f"{i * 1000 // n:03d}-{(i + 1) * 1000 // n:03d}"] = dict(
                    weighted_loss=weighted / count, unweighted_mse=unweighted / count,
                    sample_count=int(count))
        return result

    @staticmethod
    def write(writer, phase, step, report):
        writer.add_scalar(f"Loss/{phase}_interval", report["weighted_loss"], step)
        writer.add_scalar(f"MSE/{phase}_interval_unweighted", report["unweighted_mse"], step)
        writer.add_scalar(f"Counts/{phase}_interval_samples", report["sample_count"], step)
        for name, row in report["bins"].items():
            for metric, value in row.items():
                writer.add_scalar(f"Timestep/{phase}/{name}/{metric}", value, step)
