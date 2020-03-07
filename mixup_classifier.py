from mixup import *
import torch.nn as nn
from transformers import BertModel

class MixUpSentimentClassifierLayer(nn.Module):
    def __init__(self):
        super(MixUpSentimentClassifierLayer, self).__init__()
        
        #Classification layer
        self.cls_layer = nn.Linear(768, 1)
    
    def forward(self, cls_rep):
        #Feeding cls_rep to the classifier layer
        logits = self.cls_layer(cls_rep)
        return logits
    
def create_bert(additional_pretrain, finetune=False):
    BERT = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    if additional_pretrain:
        model_dict = torch.load(additional_pretrain)
        BERT.load_state_dict(model_dict)
    #Freeze bert layers
    if not finetune:
        for p in BERT.parameters():
            p.requires_grad = False
    else:
        for i, child in enumerate(BERT.encoder.layer.children()):
            if i<11:
                for p in child.parameters():
                    p.requires_grad = False
            else:
                for p in child.parameters():
                    p.requires_grad = True
        
    return BERT