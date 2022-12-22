from enum import Enum


class ModelTypes(Enum):
    PLAIN_BACKBONE = "plain_backbone"
    TUNED_WITH_CROSS_ENTROPY = "tuned_with_cross_entropy"
    TUNED_WITH_ARCFACE = "tuned_with_arcface"
    TUNED_SIAMESE_WITH_CONTRASTIVE = "tuned_siamese_with_contrastive"
    TUNED_WITH_TRIPLET = "tuned_with_triplet"
