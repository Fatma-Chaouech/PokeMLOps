from torchvision import transforms

def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(to_rgb),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return transform(image).unsqueeze(0)


def to_rgb(x):
    if x.shape[0] == 4:
        return transforms.functional.to_pil_image(x).convert('RGB')
    else:
        return transforms.functional.to_pil_image(x)

