import os
import random
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def flip(img_ir, img_vis, img_mask, boxes):
    img_ir = img_ir.transpose(Image.FLIP_LEFT_RIGHT)
    img_vis = img_vis.transpose(Image.FLIP_LEFT_RIGHT)
    img_mask = img_mask.transpose(Image.FLIP_LEFT_RIGHT)
    width = img_ir.width
    if boxes.shape[0] != 0:
        xmin = width - boxes[:, 2]
        xmax = width - boxes[:, 0]
        boxes[:, 2] = xmax
        boxes[:, 0] = xmin
    return img_ir, img_vis, img_mask, boxes


class VOCDataset(torch.utils.data.Dataset):
    CLASSES_NAME = (
        "__background__ ",
        "Lamp",
        "Car",
        "Bus",
        "Motorcycle",
        "Truck",
        "People",
    )

    def __init__(
        self,
        root_dir,
        resize_size=None,
        split="train val",
        is_train=True,
        augment=None,
    ):
        self.root = root_dir
        self.imgset = split
        self._annopath = os.path.join(self.root, "Annotation", "%s.xml")
        self._imgpath_ir = os.path.join(self.root, "Ir", "%s.png")
        self._imgpath_vis = os.path.join(self.root, "Vis", "%s.png")
        self._imgpath_mask = os.path.join(
            self.root, "Mask_DSPFusion", "%s.png"
        )
        self._imgsetpath = os.path.join(self.root, "Main", "%s.txt")

        with open(self._imgsetpath % self.imgset, encoding="utf-8") as handle:
            self.img_ids = [line.strip() for line in handle]
        self.name2id = dict(
            zip(VOCDataset.CLASSES_NAME, range(len(VOCDataset.CLASSES_NAME)))
        )
        self.resize_size = resize_size or [800, 1333]
        self.mean = [0, 0, 0]
        self.std = [1, 1, 1]
        self.train = is_train
        self.augment = augment

    def __len__(self):
        return len(self.img_ids)

    @staticmethod
    def _read_annotation(annotation_path, name2id):
        root = ET.parse(annotation_path).getroot()
        boxes = []
        classes = []
        for obj in root.iter("object"):
            box_node = obj.find("bndbox")
            box = [
                box_node.find("xmin").text,
                box_node.find("ymin").text,
                box_node.find("xmax").text,
                box_node.find("ymax").text,
            ]
            boxes.append(
                tuple(map(lambda value: value - 1, map(float, box)))
            )
            classes.append(name2id[obj.find("name").text.strip()])
        return np.asarray(boxes, dtype=np.float32), classes

    def __getitem__(self, index):
        image_id = self.img_ids[index]
        img_ir = Image.open(self._imgpath_ir % image_id).convert("L")
        img_vis = Image.open(self._imgpath_vis % image_id)
        img_mask = Image.open(self._imgpath_mask % image_id).convert("L")
        boxes, classes = self._read_annotation(
            self._annopath % image_id,
            self.name2id,
        )

        if self.train:
            if random.random() < 0.5:
                img_ir, img_vis, img_mask, boxes = flip(
                    img_ir, img_vis, img_mask, boxes
                )
            if self.augment is not None:
                img_ir, img_vis, img_mask, boxes = self.augment(
                    img_ir, img_vis, img_mask, boxes
                )

        img_ir = np.array(img_ir)
        img_vis = np.array(img_vis)
        img_mask = np.array(img_mask)
        img_ir, img_vis, img_mask, boxes = self.preprocess_img_boxes2(
            img_ir,
            img_vis,
            img_mask,
            boxes,
            self.resize_size,
        )
        return (
            transforms.ToTensor()(img_ir),
            transforms.ToTensor()(img_vis),
            transforms.ToTensor()(img_mask),
            torch.from_numpy(boxes),
            torch.LongTensor(classes),
        )

    @staticmethod
    def preprocess_img_boxes2(
        image_ir,
        image_vis,
        image_mask,
        boxes,
        input_ksize,
    ):
        min_side, max_side = input_ksize
        height, width = image_ir.shape
        image_ir = np.expand_dims(image_ir, axis=2)
        image_ir = np.concatenate((image_ir, image_ir, image_ir), axis=2)
        image_mask = np.expand_dims(image_mask, axis=2)
        image_mask = np.concatenate(
            (image_mask, image_mask, image_mask),
            axis=2,
        )

        smallest_side = min(width, height)
        largest_side = max(width, height)
        scale = min_side / smallest_side
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        new_width, new_height = int(scale * width), int(scale * height)
        image_ir = cv2.resize(image_ir, (new_width, new_height))
        image_vis = cv2.resize(image_vis, (new_width, new_height))
        image_mask = cv2.resize(image_mask, (new_width, new_height))

        pad_width = (-new_width) % 32
        pad_height = (-new_height) % 32
        image_paded_ir = np.zeros(
            shape=[new_height + pad_height, new_width + pad_width, 3],
            dtype=np.uint8,
        )
        image_paded_ir[:new_height, :new_width, :] = image_ir
        image_paded_vis = np.zeros_like(image_paded_ir)
        image_paded_vis[:new_height, :new_width, :] = image_vis
        image_paded_mask = np.zeros_like(image_paded_ir)
        image_paded_mask[:new_height, :new_width, :] = image_mask

        boxes[:, [0, 2]] *= scale
        boxes[:, [1, 3]] *= scale
        return image_paded_ir, image_paded_vis, image_paded_mask, boxes

    def collate_fn(self, data):
        (
            images_ir,
            images_vis,
            images_mask,
            boxes_list,
            classes_list,
        ) = zip(*data)
        batch_size = len(boxes_list)
        heights = [int(image.shape[1]) for image in images_ir]
        widths = [int(image.shape[2]) for image in images_ir]
        max_height = np.asarray(heights).max()
        max_width = np.asarray(widths).max()

        pad_ir = []
        pad_vis = []
        pad_mask = []
        for index in range(batch_size):
            image_ir = images_ir[index]
            pad_ir.append(
                transforms.Normalize(self.mean, self.std, inplace=True)(
                    torch.nn.functional.pad(
                        image_ir,
                        (
                            0,
                            int(max_width - image_ir.shape[2]),
                            0,
                            int(max_height - image_ir.shape[1]),
                        ),
                        value=0.0,
                    )
                )
            )

            image_vis = images_vis[index]
            pad_vis.append(
                transforms.Normalize(self.mean, self.std, inplace=True)(
                    torch.nn.functional.pad(
                        image_vis,
                        (
                            0,
                            int(max_width - image_ir.shape[2]),
                            0,
                            int(max_height - image_ir.shape[1]),
                        ),
                        value=0.0,
                    )
                )
            )

            image_mask = images_mask[index]
            pad_mask.append(
                transforms.Normalize(self.mean, self.std, inplace=True)(
                    torch.nn.functional.pad(
                        image_mask,
                        (
                            0,
                            int(max_width - image_ir.shape[2]),
                            0,
                            int(max_height - image_ir.shape[1]),
                        ),
                        value=0.0,
                    )
                )
            )

        max_targets = max(boxes.shape[0] for boxes in boxes_list)
        pad_boxes = [
            torch.nn.functional.pad(
                boxes,
                (0, 0, 0, max_targets - boxes.shape[0]),
                value=-1,
            )
            for boxes in boxes_list
        ]
        pad_classes = [
            torch.nn.functional.pad(
                classes,
                (0, max_targets - classes.shape[0]),
                value=-1,
            )
            for classes in classes_list
        ]
        return (
            torch.stack(pad_ir),
            torch.stack(pad_vis),
            torch.stack(pad_mask),
            torch.stack(pad_boxes),
            torch.stack(pad_classes),
        )
