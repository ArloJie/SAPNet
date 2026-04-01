import os
import tqdm
import sys

filepath = os.path.split(os.path.abspath(__file__))[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from eval_functions import *
from misc import *

def evaluate(name):

    pred_root = r''
    gt_root = r''

    pred_root = os.path.join(pred_root)
    gt_root = os.path.join(gt_root)

    preds = os.listdir(pred_root)
    gts = os.listdir(gt_root)
    preds = sort(preds)
    gts = sort(gts)

    preds = [i for i in preds]
    gts = [i for i in gts]

    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    MAE = Mae()

    samples = enumerate(zip(preds, gts))
    for i, sample in samples:
        pred, gt = sample

        pred_mask = np.array(Image.open(os.path.join(pred_root, pred)).convert('L'))
        gt_mask = np.array(Image.open(os.path.join(gt_root, gt)).convert('L'))

        if pred_mask.shape != gt_mask.shape:
            pred_mask = np.array(
                Image.open(os.path.join(pred_root, pred)).convert('L').resize((gt_mask.shape[1], gt_mask.shape[0])))

        FM.step(pred=pred_mask, gt=gt_mask)
        WFM.step(pred=pred_mask, gt=gt_mask)
        SM.step(pred=pred_mask, gt=gt_mask)
        EM.step(pred=pred_mask, gt=gt_mask)
        MAE.step(pred=pred_mask, gt=gt_mask)


    Sm = SM.get_results()["sm"]
    wFm = WFM.get_results()["wfm"]
    mae = MAE.get_results()["mae"]
    Em = EM.get_results()["em"]
    avgEm = Em["curve"].mean()
    maxEm = Em["curve"].max()
    Fm = FM.get_results()["fm"]
    fm_curve = Fm["curve"]
    maxFm = fm_curve.max()
    avgFm = fm_curve.mean()

    print('Sm:', Sm, 'maxEm:', maxEm, 'avgEm:', avgEm)
    print(f'F-measure:  maxFm: {maxFm:.4f} | avgFm: {avgFm:.4f}')
    print('wFm:', wFm, 'mae:', mae)


if __name__ == "__main__":
    evaluate('HKU-IS')
