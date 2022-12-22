import torch
import torch.nn as nn


class MultilabelClassifier(nn.Module):
    def __init__(self, backbone_model, num_ftrs, num_superclasses, num_classes):
        super().__init__()
        self.model = backbone_model
        self.model_wo_fc = nn.Sequential(*(list(self.model.children())[:-1]))

        self.superclass_id = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=num_ftrs, out_features=num_superclasses)
        )
        self.class_id = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=num_ftrs, out_features=num_classes)
        )

    def forward(self, x):
        x = self.model_wo_fc(x)
        x = torch.flatten(x, 1)

        return {
            'superclass_id': self.superclass_id(x),
            'class_id': self.class_id(x),
        }
