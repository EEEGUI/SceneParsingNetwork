import numpy as np


class SegmentationScore(object):
    """
    语义分割准确度
    """
    def __init__(self, n_classes=19):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class DepthEstimateScore(object):
    """
    深度估计准确度
    """
    def __init__(self):
        self.score_dict = {'a1': [],
                           'a2': [],
                           'a3': [],
                           'abs_rel': [],
                           'rmse': [],
                           'log_10': []}

    def reset(self):
        self.score_dict = {'a1': [],
                           'a2': [],
                           'a3': [],
                           'abs_rel': [],
                           'rmse': [],
                           'log_10': []}

    def update(self, label_true, label_pred):
        errors = self._compute_errors(label_true, label_pred)
        for i, key in enumerate(self.score_dict.keys()):
            self.score_dict[key].append(errors[i])

    def get_scores(self):
        scores = {}
        for key in self.score_dict.keys():
            scores[key] = np.mean(self.score_dict[key])
        return scores

    def _compute_errors(self, gt, pred):
        pred[gt <= 0] = np.nan
        gt[gt <= 0] = np.nan
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).sum()/(thresh.size - np.isnan(thresh).sum())
        a2 = (thresh < 1.25 ** 2).sum()/(thresh.size - np.isnan(thresh).sum())
        a3 = (thresh < 1.25 ** 3).sum()/(thresh.size - np.isnan(thresh).sum())

        abs_rel = np.nanmean(np.abs(gt - pred) / gt)

        rmse = np.power(gt - pred, 2)
        rmse = np.sqrt(np.nanmean(rmse))

        log_10 = np.nanmean(np.abs(np.log10(gt) - np.log10(pred)))

        return [a1, a2, a3, abs_rel, rmse, log_10]


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
