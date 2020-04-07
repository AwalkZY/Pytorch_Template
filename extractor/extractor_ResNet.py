import os

import cv2
from torch import nn
from torchvision import transforms, models
from extractor.extractor_base import BaseExtractor

normalized_item = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}


def preprocess_image(img, normalize_item):
    trans = transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(**normalize_item)])
    img = transforms.ToPILImage()(img).convert('RGB')
    img = trans(img)
    return img.float().cuda().detach()

class ResNetExtractor(BaseExtractor):
    def __init__(self, source_path, target_path, interval):
        super().__init__()
        self.model = models.resnet50(pretrained=True).cuda()
        self.extractor = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.source_path = source_path
        self.target_path = target_path
        self.interval = interval

    def extract_all(self):
        for video in os.listdir(self.source_path):
            video_path = os.path.join(self.source_path, video)
            if os.path.isfile(video_path):
                vid = video[:-4]


    def extract_item(self, *inputs):
        pass
