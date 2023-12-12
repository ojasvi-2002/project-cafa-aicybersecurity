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

from src.attacks.cafa import CaFA
from src.datasets.load_tabular_data import TabularDataset
from src.models.utils import load_trained_model



# load data: # TODO CONFIGURABLE
adult_params = dict(
        dataset_name='adult',
        data_file_path='data/adult/adult.data',
        metadata_file_path='data/adult/adult.metadata.csv',
        encoding_method='one_hot_encoding'
)
bank_params = dict(
    dataset_name='bank',
    data_file_path='data/bank/bank-full.csv',
    metadata_file_path='data/bank/bank.metadata.csv',
    encoding_method='one_hot_encoding'
)
phishing_params = dict(
    dataset_name='phishing',
    data_file_path='data/phishing/Phishing_Legitimate_full.arff',
    metadata_file_path='data/phishing/phishing.metadata.csv',
    encoding_method='one_hot_encoding'
)
data_params = adult_params
tab_dataset = TabularDataset(**data_params)
model = load_trained_model(f"{data_params['dataset_name']}-mlp.ckpt", model_type='mlp')

# TODO verify 'model'-s training data as the same properties as 'tab_dataset'

# CE Loss
def model_loss(output, target):
    output = output.float()
    target = target.long()
    return torch.functional.F.cross_entropy(output, target)


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
attack = CaFA(
    estimator=classifier,
    standard_factors=tab_dataset.standard_factors,
    cat_indices=tab_dataset.cat_indices,
    ordinal_indices=tab_dataset.ordinal_indices,
    cont_indices=tab_dataset.cont_indices,
    feature_ranges=tab_dataset.feature_ranges,
    cat_encoding_method=tab_dataset.cat_encoding_method,
    one_hot_groups=tab_dataset.one_hot_groups,
    max_iter=500,
    max_iter_tabpgd=100,
    eps=1/30,
    step_size=1/3000,
    random_init=True,

    summary_writer=False
)
X_adv = attack.generate(x=X, y=y)

# Step 7: Evaluate the ART classifier on adversarial test examples
predictions = classifier.predict(X_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == y) / len(y)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
