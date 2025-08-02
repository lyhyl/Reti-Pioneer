from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageCms

from EyeQ.MCF_Net.networks.densenet_mcf import dense121_mcs
from model.RETFound import get_RETFound_model
from model.RetiPioneer import get_reti_pioneer

srgb_profile = ImageCms.createProfile("sRGB")
lab_profile = ImageCms.createProfile("LAB")
rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")


def eyeq_prep(image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    img_hsv = image.convert("HSV")
    img_lab = ImageCms.applyTransform(image, rgb2lab_transform)
    img_rgb = np.asarray(image).astype(np.float32)
    img_hsv = np.asarray(img_hsv).astype(np.float32)
    img_lab = np.asarray(img_lab).astype(np.float32)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(img_rgb), transform(img_hsv), transform(img_lab)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fundus Images
    xl = Image.open("data/Fundus_photograph_of_normal_left_eye.jpg")
    xr = Image.open("data/Fundus_photograph_of_normal_right_eye.jpg")
    xs = [xl, xr]

    # Clinical Variables
    meta = torch.ones(1, 3, device=device)

    # Qualities
    eyeq = dense121_mcs(3)
    # https://github.com/HzFu/EyeQ
    # https://1drv.ms/u/s!ArBRrL8ao6jznU6RCbo60oStjPWZ?e=qQmzST
    eyeq_pretrain = torch.load(r"EyeQ/MCF_Net/result/DenseNet121_v3_v1.tar")
    eyeq.load_state_dict(eyeq_pretrain['state_dict'])
    eyeq.to(device)
    cs = [[v.unsqueeze(0).to(device) for v in eyeq_prep(x)] for x in xs]
    qs = [eyeq(*c)[4] for c in cs]

    # Inference
    model = get_reti_pioneer(get_RETFound_model(), 1024, 3)
    model_pretrain = torch.load(r"results/model_pretrain.pth")
    model.load_state_dict(model_pretrain['state_dict'])
    model = model.to(device)
    y: torch.Tensor = F.sigmoid(model(((cs[0][0], cs[1][0]), meta, qs)))

    print(y.detach().cpu().numpy())


if __name__ == "__main__":
    main()
