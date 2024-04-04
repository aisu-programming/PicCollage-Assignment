import dotenv
dotenv.load_dotenv(".env")
import os
import cv2
import torch
import numpy as np
import pandas as pd
# from tqdm import tqdm
from typing import Tuple  # , List
from torch.utils.data import Dataset, Subset, random_split
# from concurrent.futures import ThreadPoolExecutor, as_completed



class MyDataset(Dataset):
    def __init__(self, max_workers: int = 16) -> None:
        self.load_data(max_workers)

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        image = np.array(cv2.imread(self.inputs_filepaths[index])/255, dtype=np.float32)
        return np.transpose(image, (2,0,1)), self.truths[index]
    
    def __len__(self) -> int:
        return len(self.truths)
    
    def load_data(self, max_workers: int) -> None:
        csv = pd.read_csv(f"{os.environ['DATA_DIR']}/responses.csv")
        self.truths = np.array([ vls[1] for vls in csv.values ], dtype=np.float32)
        self.truths = np.expand_dims(self.truths, axis=-1)
        self.inputs_filepaths = [ f"{os.environ['DATA_DIR']}/images/{vls[0]}.png"
                                  for vls in csv.values ]
        # self.inputs: List[np.ndarray] = []
        # with ThreadPoolExecutor(max_workers) as executor:
        #     future_to_image = { executor.submit(cv2.imread, path): path
        #                         for path in inputs_filepaths }
        #     for future in tqdm(as_completed(future_to_image),
        #                        desc="Loading images", total=len(inputs_filepaths)):
        #         image_path = future_to_image[future]
        #         try:
        #             image = cv2.cvtColor(future.result(), cv2.COLOR_BGR2RGB)
        #             if image is not None:
        #                 image = np.transpose(image / 255, (2, 0, 1))
        #                 self.inputs.append(image)
        #             else:
        #                 print(f"Failed to load image at {image_path}")
        #         except Exception as exc:
        #             print(f"{image_path} generated an exception: {exc}")


def split_dataset(dataset: Dataset) -> Tuple[Subset, Subset]:
    dataset_length = len(dataset)
    train_dataset_length = int(dataset_length*0.8)
    valid_dataset_length = dataset_length - train_dataset_length
    train_dataset, valid_dataset = \
        random_split(
            dataset, [train_dataset_length, valid_dataset_length],
            generator=torch.Generator().manual_seed(0))
    return train_dataset, valid_dataset