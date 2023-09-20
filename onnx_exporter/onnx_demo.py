import argparse
import cv2
import yaml
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt

from typing import Any, Union, Tuple
from copy import deepcopy


parser = argparse.ArgumentParser(description="Inference an image with onnxruntime backend.")
parser.add_argument(
    "--encoder_model", type=str, required=True,
    help="Path to the efficientvit onnx encoder model."
)
parser.add_argument(
    "--decoder_model", type=str, required=True,
    help="Path to the efficientvit onnx decoder model.",
)
parser.add_argument(
    "--img_path", type=str, default='assets/fig/cat.jpg',
    help="Path to the source image",
)
parser.add_argument(
    "--out_path", type=str, default='assets/demo/onnx_efficientvit_sam_demo.jpg', 
    help="Path to the output image",
)
parser.add_argument("--mode", type=str, default="point", choices=["point", "boxes"])
parser.add_argument("--point", type=str, default=None)
parser.add_argument("--boxes", type=str, default=None)

args = parser.parse_args()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   

class SamEncoder:
    """Sam encoder model.

    In this class, encoder model will encoder the input image.

    Args:
        model_path (str): sam encoder onnx model path.
        device (str): Inference device, user can choose 'cuda' or 'cpu'. default to 'cuda'.
    """

    def __init__(self,
                 model_path: str,
                 device: str = "cpu",
                 **kwargs):
        opt = ort.SessionOptions()

        if device == "cuda":
            provider = ['CUDAExecutionProvider']
        elif device == "cpu":
            provider = ['CPUExecutionProvider']
        else:
            raise ValueError("Invalid device, please use 'cuda' or 'cpu' device.")

        print(f"loading encoder model from {model_path}...")
        self.session = ort.InferenceSession(model_path,
                                            opt,
                                            providers=provider,
                                            **kwargs)

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        self.output_shape = self.session.get_outputs()[0].shape
        self.input_size = (self.input_shape[-1], self.input_shape[-2])

    def _extract_feature(self, tensor: np.ndarray) -> np.ndarray:
        """extract image feature

        this function can use vit to extract feature from transformed image.

        Args:
            tensor (np.ndarray): input image with BGR format.

        Returns:
            np.ndarray: image`s feature.
        """
        feature = self.session.run(None, {self.input_name: tensor})[0]
        return feature

    def __call__(self, img: np.array, *args: Any, **kwds: Any) -> Any:
        return self._extract_feature(img)

class SamDecoder:
    """Sam decoder model.

    This class is the sam prompt encoder and lightweight mask decoder.

    Args:
        model_path (str): decoder model path.
        device (str): Inference device, user can choose 'cuda' or 'cpu'. default to 'cuda'.
    """

    def __init__(self,
                 model_path: str,
                 device: str = "cpu",
                 target_size: int = 1024,
                 mask_threshold: float = 0.0,
                 **kwargs):
        opt = ort.SessionOptions()

        if device == "cuda":
            provider = ['CUDAExecutionProvider']
        elif device == "cpu":
            provider = ['CPUExecutionProvider']
        else:
            raise ValueError("Invalid device, please use 'cuda' or 'cpu' device.")

        print(f"loading decoder model from {model_path}...")
        self.target_size = target_size
        self.mask_threshold = mask_threshold
        self.session = ort.InferenceSession(model_path,
                                            opt,
                                            providers=provider,
                                            **kwargs)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def run(self,
            img_embeddings: np.ndarray,
            origin_image_size: Union[list, tuple],
            point_coords: Union[list, np.ndarray] = None,
            point_labels: Union[list, np.ndarray] = None,
            boxes: Union[list, np.ndarray] = None,
            mask_input: np.ndarray = None,
            return_logits: bool = False):
        """decoder forward function

        This function can use image feature and prompt to generate mask. Must input
        at least one box or point.

        Args:
            img_embeddings (np.ndarray): the image feature from vit encoder.
            origin_image_size (list or tuple): the input image size.
            point_coords (list or np.ndarray): the input points.
            point_labels (list or np.ndarray): the input points label, 1 indicates
                a foreground point and 0 indicates a background point.
            boxes (list or np.ndarray): A length 4 array given a box prompt to the
                model, in XYXY format.
            mask_input (np.ndarray): A low resolution mask input to the model,
                typically coming from a previous prediction iteration. Has form
                1xHxW, where for SAM, H=W=4 * embedding.size.

        Returns:
            the segment results.
        """
        input_size = self.get_preprocess_shape(
            *origin_image_size, long_side_length=self.target_size
        )

        if point_coords is None and point_labels is None and boxes is None:
            raise ValueError("Unable to segment, please input at least one box or point.")

        if img_embeddings.shape != (1, 256, 64, 64):
            raise ValueError("Got wrong embedding shape!")
        if mask_input is None:
            mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            has_mask_input = np.zeros(1, dtype=np.float32)
        else:
            mask_input = np.expand_dims(mask_input, axis=0)
            has_mask_input = np.ones(1, dtype=np.float32)
            if mask_input.shape != (1, 1, 256, 256):
                raise ValueError("Got wrong mask!")
        if point_coords is not None:
            if isinstance(point_coords, list):
                point_coords = np.array(point_coords, dtype=np.float32)
            if isinstance(point_labels, list):
                point_labels = np.array(point_labels, dtype=np.float32)

        if point_coords is not None:
            point_coords = self.apply_coords(point_coords, origin_image_size, input_size).astype(np.float32)
            point_coords = np.expand_dims(point_coords, axis=0)
            point_labels = np.expand_dims(point_labels, axis=0)

        if boxes is not None:
            if isinstance(boxes, list):
                boxes = np.array(boxes, dtype=np.float32)
            assert boxes.shape[-1] == 4
            boxes = self.apply_boxes(boxes, origin_image_size, input_size).reshape((1, -1, 2)).astype(np.float32)
            box_label = np.array([[2, 3] for _ in range(boxes.shape[1] // 2)], dtype=np.float32).reshape((1, -1))

            if point_coords is not None:
                point_coords = np.concatenate([point_coords, boxes], axis=1)
                point_labels = np.concatenate([point_labels, box_label], axis=1)
            else:
                point_coords = boxes
                point_labels = box_label

        assert point_coords.shape[0] == 1 and point_coords.shape[-1] == 2
        assert point_labels.shape[0] == 1
        input_dict = {"image_embeddings": img_embeddings,
                      "point_coords": point_coords,
                      "point_labels": point_labels,
                      "mask_input": mask_input,
                      "has_mask_input": has_mask_input,
                      "orig_im_size": np.array(origin_image_size, dtype=np.float32)}
        masks, iou_predictions, low_res_masks = self.session.run(None, input_dict)

        if not return_logits:
            masks = masks > self.mask_threshold

        return masks[0], iou_predictions[0], low_res_masks[0]

    def apply_coords(self, coords, original_size, new_size):
        old_h, old_w = original_size
        new_h, new_w = new_size
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes, original_size, new_size):
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size, new_size)
        return boxes.reshape(-1, 4)

def main():
    encoder = SamEncoder(
        model_path=args.encoder_model
    )
    decoder = SamDecoder(
        model_path=args.decoder_model,
    )

    raw_image = cv2.cvtColor(cv2.imread(args.img_path), cv2.COLOR_BGR2RGB)
    img_embeddings = encoder(raw_image)
    origin_image_size = raw_image.shape[:2]

    '''Specifying a specific object with a point or bounding box'''
    if  args.mode == "point":
        H, W = origin_image_size
        point = yaml.safe_load(args.point or f"[[{W // 2},{H // 2}, {1}]]")
        point_coords = np.array([(x, y) for x, y, _ in point], dtype=np.float32)
        point_labels = np.array([l for _, _, l in point], dtype=np.float32)
        masks, _, _ = decoder.run(
            img_embeddings=img_embeddings,
            origin_image_size=origin_image_size,
            point_coords=point_coords,
            point_labels=point_labels,
        )
        plt.figure(figsize=(10,10))
        plt.imshow(raw_image)
        show_mask(masks, plt.gca())
        show_points(point_coords, point_labels, plt.gca())
        plt.savefig(args.out_path)
        print(f"Result saved in {args.out_path}")
        plt.show()
    elif  args.mode == "boxes":
        boxes = np.array(yaml.safe_load(args.boxes))
        masks, _, _ = decoder.run(
            img_embeddings=img_embeddings,
            origin_image_size=origin_image_size,
            boxes=boxes,
        )
        plt.figure(figsize=(10, 10))
        plt.imshow(raw_image)
        show_mask(masks, plt.gca())
        show_box(boxes, plt.gca())
        plt.axis('off')
        plt.savefig(args.out_path)
        print(f"Result saved in {args.out_path}")
        plt.show()
    else:
        raise NotImplementedError

if __name__ == '__main__':
    main()