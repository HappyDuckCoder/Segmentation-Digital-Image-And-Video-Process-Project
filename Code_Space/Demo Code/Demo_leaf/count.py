import os

def count_images(root):
    total = 0
    folders = {}

    for dirpath, _, filenames in os.walk(root):
        image_files = [f for f in filenames if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if image_files:
            class_name = os.path.basename(dirpath)
            count = len(image_files)
            folders[class_name] = count
            total += count

    return folders, total


if __name__ == "__main__":
    dataset_root = "dataset/train"   # đổi nếu cần

    folders, total = count_images(dataset_root)

    print("\n📌 Image count per class:")
    for cls, cnt in folders.items():
        print(f" - {cls}: {cnt}")

    print("\n🔥 Total images in dataset:", total)
