from model1.model.aurora_dataset import AuroraDataset, train_transform, eval_transform
from model1.model.cnn_model import AuroraCNN
from torch.utils.data import DataLoader

from model1.tests.unit_tests.dataset_test import test_train_dataset_single_sample, test_train_dataset_normalization_range, test_train_dataset_augmentation_variability
from model1.tests.unit_tests.model_test import test_model_forward, test_gap_removes_spatial_dims
from model1.tests.unit_tests.train_test import test_full_pipeline
from model1 import config

def main():
    #datasets with and without augmentation use different transforms
    if config.IS_128x128:
        dataset_without_augmentation = AuroraDataset(config.TRAIN_128_DIR, eval_transform)
        dataset_with_augmentation = AuroraDataset(config.TRAIN_128_DIR, train_transform)
        print("Datatset images has 128x128.")
    else:
        dataset_without_augmentation = AuroraDataset(config.TRAIN_256_DIR, eval_transform)
        dataset_with_augmentation = AuroraDataset(config.TRAIN_256_DIR, train_transform)
        print("Datatset images has 256x256.")

    dataloader = DataLoader(dataset_without_augmentation, batch_size=4)
    model = AuroraCNN()

    #dataset_without_augumentation tests
    print("Running dataset_without_augmentation test...")
    test_train_dataset_single_sample(dataset_without_augmentation)

    print("Running normalization test...")
    test_train_dataset_normalization_range(dataset_without_augmentation)

    print("Running augmentation test...")
    test_train_dataset_augmentation_variability(dataset_with_augmentation)

    #model tests
    print("Running model tests...")
    test_model_forward(model)
    test_gap_removes_spatial_dims(model)

    print("Running pipeline sanity test...")
    test_full_pipeline(dataloader, model)

    print("ALL CORE TESTS PASSED")

if __name__ == "__main__":
    main()
