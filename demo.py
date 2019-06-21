import os
import torch
import yaml
import numpy as np
import scipy.misc as misc
from models.icnet import icnet
from models.spnet import spnet
from cityscapes_loader import cityscapesLoader
import cv2

models = {'spnet': spnet, 'icnet': icnet}


def test_img(cfg):
    device = torch.device(cfg['device'])
    data_loader = cityscapesLoader
    loader = data_loader(root=cfg['data']['path'], is_transform=True, test_mode=True)
    n_classes = loader.n_classes
    # Setup Model
    if 'multi_results' in cfg['model']:
        cfg['model']['multi_results'] = False

    model = models[cfg['arch']](**cfg['model'])
    model.load_state_dict(torch.load(cfg['testing']['model_path'])["model_state"])
    model.eval()
    model.to(device)

    for img_name in os.listdir(cfg['testing']['img_fold']):
        seg_output_path = os.path.join(cfg['testing']['output_fold'], 'seg_%s.png' % img_name.split('.')[0])
        depth_output_path = os.path.join(cfg['testing']['output_fold'], 'depth_%s.png' % img_name.split('.')[0])
        if not os.path.exists(seg_output_path):
            img_path = os.path.join(cfg['testing']['img_fold'], img_name)
            img = misc.imread(img_path)
            orig_size = img.shape[:-1]

            # uint8 with RGB mode, resize width and height which are odd numbers
            # img = misc.imresize(img, (orig_size[0] // 2 * 2 + 1, orig_size[1] // 2 * 2 + 1))
            img = misc.imresize(img, (cfg['testing']['img_rows'], cfg['testing']['img_cols']))
            img = img.astype(np.float64)
            img = img[:, :, ::-1]  # RGB -> BGR
            img = img.astype(float) / 255.0
            # HWC -> CHW
            img = img.transpose(2, 0, 1)
            img = np.expand_dims(img, 0)
            img = torch.from_numpy(img).float()

            img = img.to(device)
            depth_result, seg_result = model(img)[0]

            # save segmentation result
            seg_result = np.squeeze(seg_result.data.max(1)[1].cpu().numpy(), axis=0)
            seg_result = seg_result.astype(np.float32)
            # float32 with F mode, resize back to orig_size
            seg_result = misc.imresize(seg_result, orig_size, "nearest", mode="F")

            decoded = loader.decode_segmap(seg_result)
            misc.imsave(seg_output_path, decoded)

            # save depth map
            if cfg['testing']['pred_depth']:
                depth_result = np.squeeze(depth_result.cpu().detach().numpy(), axis=0)
                depth_result = np.squeeze(depth_result, axis=0)
                depth_result = depth_result.astype(np.float32)
                # float32 with F mode, resize back to orig_size
                depth_result = misc.imresize(depth_result, orig_size, "nearest", mode='F')

                depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_result, alpha=15), cv2.COLORMAP_JET)
                # convert to mat png
                misc.imsave(depth_output_path, depth_color)


if __name__ == "__main__":
    with open('config/spnet-cityscapes.yml') as fp:
        cfg = yaml.safe_load(fp)
    test_img(cfg)
