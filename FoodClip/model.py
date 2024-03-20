import torch
import torch.nn as nn
import torchvision.models as pretrained_models



class ModifiedResNet(nn.Module):
    def __init__(self, resnet):
        super(ModifiedResNet, self).__init__()
        pretrained_model = pretrained_models.resnet50(pretrained=True)

        for param in pretrained_model.parameters():
            param.requires_grad = False

        self.pretrained_model = pretrained_model
        self.pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
        

    def forward(self, x):
        return self.resnet(x)
    
class ModifiedBert(nn.Module):
    def __init__(self, bert):
        super(ModifiedBert, self).__init__()
        self.bert = bert
        self.bert.fc = nn.Linear(768, 10)

    def forward(self, x):
        return self.bert(x)
    
class ModifiedTextTransformer(nn.Module):
    def __init__(self, text_transformer):
        super(ModifiedTextTransformer, self).__init__()
        self.text_transformer = text_transformer
        self.text_transformer.fc = nn.Linear(768, 10)

    def forward(self, x):
        return self.text_transformer(x)

class FoodClip(nn.Module):
    def __init__(self):
        super(FoodClip, self).__init__()
        self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)