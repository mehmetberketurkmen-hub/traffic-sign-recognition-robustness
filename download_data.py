from torchvision.datasets import GTSRB
import os

data_path = os.path.join(os.path.dirname(__file__), '..', 'data')

print("Downloading GTSRB Dataset...")
print("This process may take a few minutes depending on your internet speed, please wait.\n")


train_data = GTSRB(root=data_path, split='train', download=True)
test_data = GTSRB(root=data_path, split='test', download=True)

print("Success! Dataset downloaded to the 'data' folder.")
print(f"Total Training Images: {len(train_data)}")
print(f"Total Test Images: {len(test_data)}")