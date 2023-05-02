from torchvision import transforms
import torch

def denormalize(image):
    image = (image * torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)) + torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    image = torch.clamp(image, 0, 1)
    
    transform = transforms.Compose([
        transforms.ToPILImage(mode='RGB')
    ])
    return transform(image)