import os
import torch
import numpy as np
import time  
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from torch.utils.data import Dataset, DataLoader
from Networks.model1 import Model1
from Networks.model2 import Model2
from Networks.model3 import Model3


class LowLightDataset(Dataset):
    def __init__(self, low_light_dir, transform=None):
        self.low_light_dir = low_light_dir
        self.low_light_images = sorted(os.listdir(low_light_dir))
        self.transform = transform
    
    def __len__(self):
        return len(self.low_light_images)
    
    def __getitem__(self, idx):
        low_light_img_path = os.path.join(self.low_light_dir, self.low_light_images[idx])
        
        low_light_img = Image.open(low_light_img_path).convert("RGB")
        
        if self.transform:
            low_light_img = self.transform(low_light_img)
        
        return low_light_img


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model_restoration = Model1()  # Choose the specific model architecture you want to use (Model1 in this case)

# Move the model to the device (GPU/CPU)
model_restoration = model_restoration.to(device)  # Ensure that the model is on the correct device (GPU/CPU)

# Load the trained model weights from the checkpoint
checkpoint = torch.load('trained/model1.pth')  # Load the pre-trained weights stored in the checkpoint file

# Load the model's state dictionary (weights) into the model
model_restoration.load_state_dict(checkpoint['state_dict'])  # Load the model's trained weights from the checkpoint

model_restoration.eval()

low_light_dir = './Input/Low'
transform = Compose([
    Resize((352, 1216)),
    CenterCrop((352, 1216)),
    ToTensor()
])

dataset = LowLightDataset(low_light_dir, transform=transform)
test_loader = DataLoader(dataset, batch_size=2, shuffle=False)

save_images = True
result_dir = './Result'

if save_images:
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    resize_transform = Resize((376, 1241))
    j = 0
    
    total_time = 0  # To accumulate the total processing time
    total_images = 0  # To count the total number of images processed

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            print("Processing image: ", i)
            start_time = time.time()  # Start timing

            input_ = data.to(device)
            restored = model_restoration(input_)
            restored = torch.clamp(restored, 0, 1)
            
            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
            
            batch_size = restored.shape[0]
            total_images += batch_size
            
            for batch in range(batch_size):
                restored_image = (restored[batch] * 255).astype(np.uint8)
                restored_image = Image.fromarray(restored_image)
                restored_image = resize_transform(restored_image)
                
                restored_image.save(os.path.join(result_dir, f'{j:06d}.jpg'))
                j += 1

            end_time = time.time()  # End timing
            batch_time = end_time - start_time  # Calculate time for the batch
            total_time += batch_time  # Accumulate total processing time

    avg_time_per_image = total_time / total_images if total_images > 0 else 0
    print(f"Average processing time per image: {avg_time_per_image:.4f} seconds")

