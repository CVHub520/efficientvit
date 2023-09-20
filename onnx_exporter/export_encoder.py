import argparse
import warnings
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize

from efficientvit.sam_model_zoo import create_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor

parser = argparse.ArgumentParser(
    description="Export the efficient-sam encoder to an onnx model."
)
parser.add_argument(
    "--checkpoint", type=str, required=True,
    help="The path to the efficient-sam model checkpoint.",
)
parser.add_argument(
    "--output", type=str, required=True,
    help="The filename to save the onnx model to.",
)
parser.add_argument(
    "--model-type", type=str, required=True,
    help="In ['l0', 'l1'], Which type of efficient-sam model to export.",
)
parser.add_argument(
    "--opset", type=int, default=12,
    help="The ONNX opset version to use. Must be >=11",
)
parser.add_argument(
    "--use-preprocess", action="store_true",
    help=("Embed pre-processing into the model",),
)
parser.add_argument(
    "--quantize-out", type=str, default=None,
    help=(
        "If set, will quantize the model and save it with this name. "
        "Quantization is performed with quantize_dynamic from "
        "onnxruntime.quantization.quantize."
    ),
)
parser.add_argument(
    "--gelu-approximate",
    action="store_true",
    help=(
        "Replace GELU operations with approximations using tanh. Useful "
        "for some runtimes that have slow or unimplemented erf ops, used in GELU."
    ),
)

class SamResize:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        h, w, _ = image.shape
        long_side = max(h, w)
        if long_side != self.size:
            return self.apply_image(image)
        else:
            return image

    def apply_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects a torch tensor with shape HxWxC in float format.
        """
        h, w, _ = image.shape
        long_side = max(h, w)
        if long_side != self.size:
            target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.size)
            x = resize(image.permute(2, 0, 1), target_size)
            return x
        else:
            return image.permute(2, 0, 1)

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

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size})"


class EncoderModel(nn.Module):
    """
    This model should not be called directly, but is used in ONNX export.
    It combines the image encoder of Sam, with some functions modified to enable model tracing. 
    Also supports extra options controlling what information. 
    See the ONNX export script for details.
    """

    def __init__(
        self,
        predictor: EfficientViTSamPredictor,
        use_preprocess: bool,
        pixel_mean: List[float] = [123.675 / 255, 116.28 / 255, 103.53 / 255],
        pixel_std: List[float] = [58.395 / 255, 57.12 / 255, 57.375 / 255],
    ):
        super().__init__()

        self.pixel_mean = torch.tensor(pixel_mean, dtype=torch.float)
        self.pixel_std = torch.tensor(pixel_std, dtype=torch.float)

        self.model = predictor.model
        self.image_size = predictor.model.image_size
        self.image_encoder = self.model.image_encoder
        self.use_preprocess = use_preprocess
        self.resize_transform = SamResize(size=self.model.image_size[1])
        self.transform = self.model.transform

    @torch.no_grad()
    def forward(self, input_image):
        if self.use_preprocess:
            input_image = self.preprocess(input_image)
        image_embeddings = self.image_encoder(input_image)
        return image_embeddings

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        
        # Resize & Permute to (C,H,W)
        x = self.resize_transform(x)

        # Normalize
        x = x.float() / 255
        x = transforms.Normalize(mean=self.pixel_mean, std=self.pixel_std)(x)

        # Pad
        h, w = x.shape[-2:]
        th, tw = self.image_size[1], self.image_size[1]
        assert th >= h and tw >= w
        padh = th - h
        padw = tw - w
        x = F.pad(x, (0, padw, 0, padh), value=0)

        # Expand
        x = torch.unsqueeze(x, 0)

        return x


def run_export(
    model_type: str,
    checkpoint: str,
    output: str,
    use_preprocess: bool,
    opset: int,
    gelu_approximate: bool = False,
) -> None:
    print("Loading model...")
    # build model
    efficientvit_sam = create_sam_model(model_type, True, checkpoint).eval()
    efficientvit_sam_predictor = EfficientViTSamPredictor(efficientvit_sam)

    onnx_model = EncoderModel(
        predictor=efficientvit_sam_predictor,
        use_preprocess=use_preprocess,
    )

    if gelu_approximate:
        for _, m in onnx_model.named_modules():
            if isinstance(m, torch.nn.GELU):
                m.approximate = "tanh"

    image_size = [1944, 2592]
    if use_preprocess:
        dummy_input = {
            "input_image": torch.randint(
                0, 255, (image_size[0], image_size[1], 3), dtype=torch.uint8
            )
        }
        dynamic_axes = {
            "input_image": {0: "image_height", 1: "image_width"},
        }
    else:
        dummy_input = {
            "input_image": torch.randn(
                (1, 3, image_size[0], image_size[1]), dtype=torch.float
            )
        }
        dynamic_axes = None
    _ = onnx_model(**dummy_input)

    output_names = ["image_embeddings"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        print(f"Exporting onnx model to {output}...")
        with open(output, "wb") as f:
            torch.onnx.export(
                onnx_model,
                tuple(dummy_input.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=opset,
                do_constant_folding=True,
                input_names=list(dummy_input.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )


if __name__ == "__main__":
    args = parser.parse_args()
    run_export(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        output=args.output,
        use_preprocess=args.use_preprocess,
        opset=args.opset,
        gelu_approximate=args.gelu_approximate,
    )

    if args.quantize_out is not None:
        from onnxruntime.quantization import QuantType  # type: ignore
        from onnxruntime.quantization.quantize import quantize_dynamic  # type: ignore

        print(f"Quantizing model and writing to {args.quantize_out}...")
        quantize_dynamic(
            model_input=args.output,
            model_output=args.quantize_out,
            optimize_model=True,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
        )
        print("Done!")
