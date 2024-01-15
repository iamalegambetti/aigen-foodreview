import torch 
import torch.nn as nn

class CLIPDetector(nn.Module):
    def __init__(self, backbone, processor, out_dim=1):
        super(CLIPDetector, self).__init__()
        self.backbone = backbone
        self.processor = processor
        self.fc1 = nn.Linear(1024, out_dim) 

    def feature_extractor(self, inputs):
        outputs = self.backbone(**inputs)
        image_embeds, text_embeds = outputs.image_embeds, outputs.text_embeds
        return text_embeds, image_embeds
    
    def forward(self, inputs):
        text_embeds, image_embeds = self.feature_extractor(inputs)
        fused = torch.cat([image_embeds, text_embeds], dim = 1) # this fusion is just a simple concatenation 
        output = self.fc1(fused)
        return output
    
class FLAVADetector(nn.Module):
    def __init__(self, backbone, processor, out_dim=1):
        super(FLAVADetector, self).__init__()
        self.backbone = backbone
        self.processor = processor
        self.fc1 = nn.Linear(768, out_dim) 

    def feature_extractor(self, inputs):
        outputs = self.backbone(**inputs)
        embeddings = outputs.multimodal_embeddings
        cls_embedding = embeddings[:, 0, :]
        return cls_embedding
    
    def forward(self, inputs):
        x = self.feature_extractor(inputs)
        return self.fc1(x)