from torch import nn
from torchvision import models
import os
from extractor.extractor_base import BaseExtractor


class R3DExtractor(BaseExtractor):
    def __init__(self, source_path, target_path):
        super().__init__()
        self.model = models.video.r2plus1d_18(pretrained=True).cuda()
        self.extractor = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.source_path = source_path
        self.target_path = target_path

    def extract_all(self):
        for dir in os.listdir(self.source_path):
            pass


    def extract_item(self, *inputs):
        pass
