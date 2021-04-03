import os

from PIL import Image
from torch.utils.data import DataLoader, Dataset


class COCODataset(Dataset):
    def __init__(self, mode, root, transform=None):
        assert mode in {"train", "val"}
        super().__init__()
        self.mode = mode
        self.data_path = os.path.join(root, f"{mode}2014")
        self.len = len(os.listdir(self.data_path))
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        title = f"COCO_{self.mode}2014_{idx:012}.jpg"
        image_path = os.path.join(self.data_path, title)
        image = Image.open(image_path)
        if self.transform is not None:
            return self.transform(image)
        return image


class VQADataset(Dataset):
    def __init__(self, mode, root, transform=None):
        assert mode in {"train", "test"}
        super().__init__()
        data_path = os.path.join(root, mode)
        self.image_ids = fetch_file(os.path.join(data_path, "img_ids.txt"))
        self.answers = fetch_file(os.path.join(data_path, "answers.txt"))
        self.questions = fetch_file(os.path.join(data_path, "questions.txt"))
        self.types = fetch_file(os.path.join(data_path, "types.txt"))
        coco_mode = "train" if mode == "train" else "val"
        self.coco_dataset = COCODataset(coco_mode, transform)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = int(self.image_ids[idx])
        image = self.coco_dataset[image_id]
        question = self.questions[idx]
        answer = self.answers[idx]
        type_ = self.types[idx]
        return question, image, answer, type_


def fetch_file(path):
    with open(path) as f:
        return f.read().splitlines()
