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
from src.utils import evaluate_crafted_samples

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
    max_iter=250,
    max_iter_tabpgd=100,
    eps=1 / 30,
    step_size=1 / 3000,
    random_init=True,

    summary_writer=False
)

X_adv = attack.generate(x=X, y=y)

# Step 7: Evaluate the ART classifier on adversarial test examples
predictions = classifier.predict(X_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == y) / len(y)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))

# Step 8: Project adversarial -----------------------------------------------------------------------------------------
from src.constraints.utilizing.constrainer import DCsConstrainer

# Initialize the dataset used for mining
adult_mining_source_params = dict(
    dataset_name='adult',
    data_file_path='data/adult/adult.data',
    metadata_file_path='data/adult/adult.metadata.csv',
    encoding_method=None
)
tab_dcs_dataset = TabularDataset(**adult_mining_source_params)

# Initialize the DCs constrainer
constrainer = DCsConstrainer(
    x_tuples_df=tab_dcs_dataset.x_df,
    eval_csv_out_path='data/adult/adult-13cols__DC_threshold=0.01__fastadc_dcs__eval.csv',
    feature_names=tab_dcs_dataset.feature_names,
    is_feature_ordinal=tab_dcs_dataset.is_feature_ordinal,
    is_feature_continuous=tab_dcs_dataset.is_feature_continuous,
    feature_types=tab_dcs_dataset.feature_types,
    feature_ranges=tab_dcs_dataset.feature_ranges,
    feature_names_dcs_format=tab_dcs_dataset.feature_names_dcs_format,
    standard_factors=tab_dcs_dataset.standard_factors,
    n_dcs=3000,
    n_tuples=1,
    # limit_cost_ball  # TODO examine
)

from src.constraints.constraint_projector import ConstraintProjector

projector = ConstraintProjector(
    constrainer=constrainer,
    upper_projection_budget_bound=0.5,
)

# collect sample projected to numpy array
X_adv_proj = []

for x_orig, x_adv in zip(X, X_adv):  # for validation

    # 1. Transform sample to the format of the DCs dataset
    sample_orig = TabularDataset.cast_sample_format(x_orig, from_dataset=tab_dataset, to_dataset=tab_dcs_dataset)
    sample_adv = TabularDataset.cast_sample_format(x_adv, from_dataset=tab_dataset, to_dataset=tab_dcs_dataset)

    # 1.1. Sanity checks:
    assert np.all(x_orig ==
                  TabularDataset.cast_sample_format(sample_orig, from_dataset=tab_dcs_dataset, to_dataset=tab_dataset))
    assert np.all(sample_orig ==
                  TabularDataset.cast_sample_format(x_orig, from_dataset=tab_dataset, to_dataset=tab_dcs_dataset))

    # 2. Project
    is_succ, sample_projected = projector.project(sample_adv, sample_original=sample_orig)

    # 3. Transform back to the format of the model input
    x_adv_proj = TabularDataset.cast_sample_format(sample_projected, from_dataset=tab_dcs_dataset,
                                                   to_dataset=tab_dataset)
    X_adv_proj.append(x_adv_proj)

X_adv_proj = np.array(X_adv_proj)

# Step 9: Evaluate the attack -----------------------------------------------------------------------------------------
eval_params = dict(
    classifier=classifier,
    constrainer=constrainer,
    tab_dataset_constrainer=tab_dcs_dataset,
    tab_dataset=tab_dataset,
)
eval_orig_samples = evaluate_crafted_samples(X_adv=X, X_orig=X, y=y, **eval_params)
print("before attack:", eval_orig_samples)
eval_adv_samples = evaluate_crafted_samples(X_adv=X_adv, X_orig=X, y=y, **eval_params)
print("after cafa:", eval_adv_samples)
eval_adv_proj_samples = evaluate_crafted_samples(X_adv=X_adv_proj, X_orig=X, y=y, **eval_params)
print("after projection:", eval_adv_proj_samples)  # TODO fix bug: currently the metrics are similar
