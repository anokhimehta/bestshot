import os
import cv2
import numpy as np
import swiftclient
from dotenv import load_dotenv
import tempfile

load_dotenv()

# Connect to object storage
conn = swiftclient.Connection(
    auth_version='3',
    authurl=os.environ['OS_AUTH_URL'],
    os_options={
        'application_credential_id': os.environ['OS_APPLICATION_CREDENTIAL_ID'],
        'application_credential_secret': os.environ['OS_APPLICATION_CREDENTIAL_SECRET'],
        'region_name': os.environ['OS_REGION_NAME'],
        'auth_type': 'v3applicationcredential'
    }
)

BUCKET = os.environ.get('BUCKET_NAME', 'ak12754-data-proj19')
IMAGE_DIR = os.environ.get('IMAGE_DIR', '/tmp/koniq_images/512x384')

def apply_blur(image):
    """Simulate blurry photo"""
    kernel_size = np.random.choice([5, 7, 9])
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_overexposure(image):
    """Simulate overexposed photo"""
    factor = np.random.uniform(1.5, 2.0)
    return np.clip(image * factor, 0, 255).astype(np.uint8)

def apply_underexposure(image):
    """Simulate underexposed photo"""
    factor = np.random.uniform(0.3, 0.6)
    return np.clip(image * factor, 0, 255).astype(np.uint8)

def apply_noise(image):
    """Simulate grainy photo"""
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    return np.clip(image + noise, 0, 255).astype(np.uint8)

def main():
    # Pick 1000 random images to augment
    images = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
    selected = np.random.choice(images, 1000, replace=False)

    augmentations = [
        ('blur', apply_blur),
        ('overexposed', apply_overexposure),
        ('underexposed', apply_underexposure),
        ('noisy', apply_noise)
    ]

    total = len(selected) * len(augmentations)
    count = 0

    print(f"Generating {total} synthetic images...")

    for filename in selected:
        filepath = os.path.join(IMAGE_DIR, filename)
        image = cv2.imread(filepath)

        if image is None:
            continue

        for aug_name, aug_func in augmentations:
            augmented = aug_func(image)
            new_filename = f"{aug_name}_{filename}"

            # Save to temp file and upload
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                cv2.imwrite(tmp.name, augmented)
                with open(tmp.name, 'rb') as f:
                    conn.put_object(
                        BUCKET,
                        f'koniq10k/synthetic/{new_filename}',
                        f
                    )
            os.unlink(tmp.name)
            count += 1

            if count % 100 == 0:
                print(f"Generated {count}/{total} synthetic images...")

    print(f"Done! Generated {total} synthetic images in koniq10k/synthetic/")

if __name__ == "__main__":
    main()
  
