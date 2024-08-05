import os
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa

def augment_and_save_images(source_folder, destination_folder, augmentations):
    print("Starting the augmentation process...")
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"Created directory: {destination_folder}")

    for filename in os.listdir(source_folder):
        if filename.lower().endswith(".jpg"):
            card_name = os.path.splitext(filename)[0]
            card_folder = os.path.join(destination_folder, card_name)
            if not os.path.exists(card_folder):
                os.makedirs(card_folder)
                print(f"Created directory: {card_folder}")
            
            img_path = os.path.join(source_folder, filename)
            print(f"Processing file: {img_path}")
            try:
                img = np.array(Image.open(img_path))
                print(f"Loaded image: {img_path}")

                # save the original image
                original_img_path = os.path.join(card_folder, f"{card_name}_original.jpg")
                Image.fromarray(img).save(original_img_path)
                print(f"Saved original image: {original_img_path}")

                # apply augs and save the modified images
                for i, aug in enumerate(augmentations):
                    aug_img = aug(image=img)
                    aug_name = f"{card_name}_aug_{i}"
                    aug_img_pil = Image.fromarray(aug_img)
                    aug_img_pil.save(os.path.join(card_folder, f"{aug_name}.jpg"))
                    print(f"Saved augmented image: {os.path.join(card_folder, f'{aug_name}.jpg')}")
            except Exception as e:
                print(f"Failed to process file {img_path}: {e}")

def get_augmentations():
    return [
        iaa.Multiply((0.5, 1.5), name="brightness"),  # vary brightness
        iaa.GaussianBlur(sigma=(0.0, 3.0), name="blur"),  # gaussian blur
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255), name="noise"),  # add Gaussian noise
        iaa.MotionBlur(k=15, name="motion_blur"),  # simulate motion blur
        iaa.LinearContrast((0.5, 2.0), name="contrast"),  # adjust contrast
        iaa.Affine(rotate=(-20, 20), name="rotate"),  # rotate
        iaa.Affine(scale=(0.5, 1.5), name="scale"),  # scale
        iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, name="translate")  # Translate
    ]

source_folder = os.path.expanduser("~/Desktop/cards")
destination_folder = os.path.expanduser("~/Desktop/editedCards")

augmentations = get_augmentations()
augment_and_save_images(source_folder, destination_folder, augmentations)
print("Augmentation process completed.")
