import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.optim import lr_scheduler

from src.configs import *
from src.models.model_types import ModelTypes
from src.models.multilabel_classifier import MultilabelClassifier


class ResNetInitializer:
    def __init__(self, model_type, num_superclasses, num_classes):
        # Load the pretrained model
        backbone_model = models.resnet18(pretrained=True)
        criterion = None

        num_ftrs = backbone_model.fc.in_features
        if model_type == ModelTypes.PLAIN_BACKBONE:
            model_name = 'plain_resnet18'
            # Here the size of each output sample is set to num_superclasses.
            backbone_model.fc = nn.Linear(num_ftrs, num_superclasses)
            model = backbone_model
        elif model_type == ModelTypes.TUNED_WITH_CROSS_ENTROPY:
            model_name = 'resnet18_with_cross_entropy_loss'
            criterion = nn.CrossEntropyLoss()
            model = MultilabelClassifier(backbone_model, num_ftrs,
                                         num_superclasses, num_classes)

        self.model = model.to(DEVICE)
        self.criterion = criterion
        self.model_name = model_name

        # Observe that all parameters are being optimized
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

    def get_model_name(self):
        return self.model_name
