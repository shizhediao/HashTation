import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class BERT_Model(nn.Module):
    def __init__(self, num_classes):
        super(BERT_Model, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
        self.model.classifier = nn.Linear(768, num_classes, bias=True)
        self.model.num_labels = num_classes

    def forward(self, x, mask):
        out = self.model(x, mask)
        return out['logits']

class ClassificationModel(nn.Module):
    def __init__(self, num_classes, frozen=False, model='timelms'):
        super(ClassificationModel, self).__init__()
        if model=="timelms":
            self.model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-mar2022')
        elif model=="bertweet":
            self.model = AutoModelForSequenceClassification.from_pretrained('vinai/bertweet-large')
        elif model=="bert":
            self.model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
        elif model=="bert-large":
            self.model = AutoModelForSequenceClassification.from_pretrained('bert-large-uncased')
        elif model=="roberta":
            self.model = AutoModelForSequenceClassification.from_pretrained('roberta-base')
        elif model=="roberta-large":
            self.model = AutoModelForSequenceClassification.from_pretrained('roberta-large')
        else:
            assert(model in ["timelms", "bertweet", "bert", "bert-large", "roberta", "roberta-large"])
        self.model.num_labels = num_classes
        self.model.classifier.out_proj = nn.Linear(768, num_classes, bias=True)

        if frozen:
            for param in self.model.roberta.parameters():
                param.requires_grad = False

    def forward(self, x, mask):
        out = self.model(x, mask)
        return out['logits']