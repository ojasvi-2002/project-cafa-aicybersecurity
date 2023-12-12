# inspiration #1: LLm attacks: https://github.com/llm-attacks/llm-attacks/tree/main/llm_attacks
# inspiration #2: ART


# stage 1: wrap classifier with ART
"""
The script demonstrates a simple example of using ART with PyTorch. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import torch
import numpy as np

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist

from src.attacks.tabpgd import TabPGD
from src.datasets.load_tabular_data import TabularDataset
from src.models.utils import load_trained_model

model = load_trained_model('epoch=26-val_hp_metric=0.825-one-hot.ckpt', model_type='mlp')

# TODO verify trainset is the same as the one used in training (for disjoint train/test)
# hyperparameters['data_summary'] = features_metadata.summary

# load data:
data_parameters = dict(dataset_name='adult',
                           data_file_path='data/adult/adult.data',
                           metadata_file_path='data/adult/adult.metadata.csv',
                           encoding_method='one_hot_encoding')  # TODO CONFIG
tab_dataset = TabularDataset(**data_parameters)


# ce loss hack
def model_loss(output, target):
    output = output.float()
    target = target.long()
    return model.loss(output, target)


# Step 3: Create the ART classifier
classifier = PyTorchClassifier(
    model=model,
    loss=model_loss,
    input_shape=tab_dataset.n_features,
    nb_classes=tab_dataset.n_classes,
)

# Step 5: Evaluate the ART classifier on benign test examples
X, y = tab_dataset.X_test[:500], tab_dataset.y_test[:500]
predictions = classifier.predict(X)
accuracy = np.sum(np.argmax(predictions, axis=1) == y) / len(y)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

# Step 6: Generate adversarial test examples
# attack = FastGradientMethod(estimator=classifier, eps=5)
attack = TabPGD(
    estimator=classifier,
    standard_factors=tab_dataset.standard_factors,
    cat_indices=tab_dataset.cat_indices,
    ordinal_indices=tab_dataset.ordinal_indices,
    cont_indices=tab_dataset.cont_indices,
    feature_ranges=tab_dataset.feature_ranges,
    cat_encoding_method=tab_dataset.cat_encoding_method,
    one_hot_groups=tab_dataset.one_hot_groups,
    max_iter=100,
    step_size=0.003,
    random_init=True,
)
X_adv = attack.generate(x=X, y=y)

# Step 7: Evaluate the ART classifier on adversarial test examples
predictions = classifier.predict(X_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == y) / len(y)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
