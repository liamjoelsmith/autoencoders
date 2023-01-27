from imports import *
import os

"""
A custom image loader dataset
"""
class ImageLoader(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_dir = img_dir

        self.total_images = os.listdir(os.path.join(self.img_dir, "images"))
        self.total_images.sort()

        self.transform = transform

    def __len__(self):
        return len(self.total_images)

    def __getitem__(self, index):
        transform = torchvision.transforms.ToTensor()
        image = os.path.join(self.img_dir, "images", self.total_images[index])
        image = transform(Image.open(image))
        if self.transform:
            image = self.transform(image)
        return image

"""
Load in image(s) and transform data via custom ImageLoader, and create our DataLoader
"""
def load_train_dataset(batch_size=8, image_resize=256, relative_path="ISBI_dataset\\"):
    transform = Compose([
                    ToPILImage(),
                    Grayscale(),  # ensure images only have one channel
                    Resize(image_resize),  # ensure all images have same size
                    CenterCrop(image_resize),
                    ToTensor(),
                ])

    # load data, with the above transform applied
    ims = ImageLoader(relative_path + "test", transform=transform)
    return DataLoader(ims, batch_size=batch_size, shuffle=False, num_workers=1)