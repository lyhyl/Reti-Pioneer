from typing import Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageCms

from model.base import get_EyeQ, get_RETFound, get_SwinB, get_VimS


class AnalysisModel:
    _instance = None
    
    @classmethod
    def get_default(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self):
        if not torch.cuda.is_available():
            raise RuntimeError("Require CUDA to run.")
        self.device = torch.device("cuda")

        self.diseases = {
            "T2D": "t2dm", # note: t2dm in table
            "Hypertension": "hypertension",
            "Hyperlipidemia": "hyperlipemia", # note: misspell
            "Gout": "gout",
            "Osteoporosis": "osteoporosis",
            "Thyroid": "thyroid",
        }

        self.threshold = {
            "T2D": 0.072917238,
            "Thyroid": 0.098484434,
            "Osteoporosis": 0.029431015,
            "Gout": 0.020402882,
            "Hyperlipidemia": 0.458788097,
            "Hypertension": 0.425803155,
        }

        self.srgb_profile = ImageCms.createProfile("sRGB")
        self.lab_profile = ImageCms.createProfile("LAB")
        self.rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(
            self.srgb_profile, self.lab_profile, "RGB", "LAB"
        )

        self.eyeq = get_EyeQ()
        self.eyeq = self.eyeq.to(self.device)
        self.eyeq.eval()

        self.retf = get_RETFound()
        self.retf = self.retf.to(self.device)
        self.retf.eval()

        self.swinb = get_SwinB()
        self.swinb = self.swinb.to(self.device)
        self.swinb.eval()

        self.vims = get_VimS()
        self.vims = self.vims.to(self.device)
        self.vims.eval()

        self.longitudinal_years = [5, 10]
        self.model = {}
        for y in [0, 5, 10]:
            for disease in self.diseases.keys():
                name = f"{disease}_y{y}"
                self.model[name] = torch.jit.load(f"models/{name}.pt")
                self.model[name].cuda()
                self.model[name].eval()

    def eyeq_prep(
        self, image: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_rgb = image.convert("RGB")
        img_hsv = image.convert("HSV")
        img_lab = ImageCms.applyTransform(image, self.rgb2lab_transform)
        if img_lab.mode != "RGB" and img_lab.mode != "LAB":
            img_lab = img_lab.convert("RGB")
        
        base_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        tensor_rgb = base_transform(img_rgb)
        tensor_hsv = base_transform(img_hsv)
        tensor_lab = base_transform(img_lab)

        if tensor_rgb.shape[0] != 3:
            tensor_rgb = tensor_rgb[:3] if tensor_rgb.shape[0] > 3 else tensor_rgb
        if tensor_hsv.shape[0] != 3:
            tensor_hsv = tensor_hsv[:3] if tensor_hsv.shape[0] > 3 else tensor_hsv
        if tensor_lab.shape[0] != 3:
            tensor_lab = tensor_lab[:3] if tensor_lab.shape[0] > 3 else tensor_lab
        
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        tensor_rgb = normalize(tensor_rgb)
        tensor_hsv = normalize(tensor_hsv)
        tensor_lab = normalize(tensor_lab)
        
        return tensor_rgb, tensor_hsv, tensor_lab
    
    def cross_sectional_inference(self, disease_name, meta, cs, qualities):
        left_features = []
        right_features = []
        for backbone in [self.retf, self.swinb, self.vims]:
            left_features.append(backbone(cs[0][0]))
            right_features.append(backbone(cs[1][0]).flip([-1]))
        logit = self.model[f"{disease_name}_y0"](((left_features, right_features), meta, qualities))
        return F.sigmoid(logit).item()

    def longitudinal_inference(self, disease_name, y, meta, cs, qualities):
        left_features = self.retf(cs[0][0])
        right_features = self.retf(cs[1][0].flip([-1]))
        logit = self.model[f"{disease_name}_y{y}"](((left_features, right_features), meta, qualities))
        return F.sigmoid(logit).item()

    def inference(self, disease_name, meta, ext_meta, cs, qualities):
        cross_prob = self.cross_sectional_inference(disease_name, torch.cat([meta, ext_meta], 1), cs, qualities)
        long_probs = [self.longitudinal_inference(disease_name, y, meta, cs, qualities) for y in self.longitudinal_years]
        return cross_prob, long_probs
    
    @torch.no_grad()
    def predict(self, req):
        """
        req:
        {
            "disease_name": "T2D", // T2D, Hypertension, Hyperlipidemia, Gout, Osteoporosis, Thyroid
            "age": 28, // float
            "gender": 0, // 0-female 1-male
            "weight": 60, // float
            "left_eye_image_path": "/home/dell/1733193497348570043.png",
            "right_eye_image_path": "/home/dell/1733193500273477447.png",
            "race": 1, // int
        }

        return:
        {
            "cross_prob": 0.5, // float
            "is_suspected": True, // bool
            "long_probs": 0.5, // list[float]
        }
        """
        disease_name = req["disease_name"]
        age = req["age"]
        gender = req["gender"]
        weight = req["weight"]
        left_eye_image_path = req["left_eye_image_path"]
        right_eye_image_path = req["right_eye_image_path"]
        race = int(req["race"])

        # Fundus Images
        xl = Image.open(left_eye_image_path)
        xr = Image.open(right_eye_image_path)
        xs = [xl, xr]

        # Clinical Variables
        eth = [0 for _ in range(7)] # one hot
        eth[race - 1] = 1
        ext_meta = torch.tensor([eth], device=self.device)

        meta = torch.tensor([[age, gender, weight]], device=self.device)
        if race != 7:
            mmean = torch.tensor([56.6064740, 0.5, 78.04054385447546], device=self.device)
            mstd = torch.tensor([8.15334413, 0.5, 15.985435090239642], device=self.device)
        elif race == 7:
            mmean = torch.tensor([63.22984,  0.5, 61.46494], device=self.device)
            mstd = torch.tensor([16.57992,  0.5, 12.420399], device=self.device)
        meta = (meta - mmean) / mstd

        # Qualities
        cs = [[v.unsqueeze(0).to(self.device) for v in self.eyeq_prep(x)] for x in xs]
        qualities = [self.eyeq(*c)[4] for c in cs]

        # Inference
        cross_prob, long_probs = self.inference(disease_name, meta, ext_meta, cs, qualities)
        is_suspected = cross_prob > self.threshold[disease_name]
        result = {
            "cross_prob": cross_prob,
            "is_suspected": is_suspected,
            "long_probs": dict(zip(self.longitudinal_years, long_probs)),
        }
        return result


if __name__ == "__main__":
    analysis_model = AnalysisModel.get_default()
    analysis_model.initialize()
    
    req = {
        "disease_name": "T2D",
        "age": 28,
        "gender": 0,
        "weight": 60,
        "race": 1,
        "left_eye_image_path": "./data/test-L.JPG",
        "right_eye_image_path": "./data/test-R.JPG",
    } 
    predict_res = analysis_model.predict(req)
    print("predict done", predict_res)
