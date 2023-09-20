#!/bin/bash

option=$1


if [ "$option" == "demo_torch" ]
then
    echo "export demo sam >>>"
    python demo_sam_model.py \
        --model l1 \
        --weight_url assets/checkpoints/sam/l1.pt \
        --mode point \
        --image_path assets/fig/cat.jpg \
        --output_path assets/demo/efficientvit_sam_demo_point.png

elif [ "$option" == "encoder" ]
then
    echo "export efficientvit sam encoder >>>"
    python -m onnx_exporter.export_encoder \
        --checkpoint assets/checkpoints/sam/l1.pt \
        --output assets/checkpoints/sam/efficientvit_sam_l1_vit_h.encoder.onnx \
        --model-type l1 \
        --opset 12 \
        --use-preprocess

elif [ "$option" == "decoder" ]
then
    echo "export efficientvit sam decoder >>>"
    python -m onnx_exporter.export_decoder \
        --checkpoint assets/checkpoints/sam/l1.pt \
        --output assets/checkpoints/sam/efficientvit_sam_l1_vit_h.decoder.onnx \
        --model-type l1 \
        --opset 12 \
        --return-single-mask

elif [ "$option" == "demo_onnx" ]
then
    echo "inference efficientvit sam using onnxruntime >>>"
    python -m onnx_exporter.onnx_demo \
    --encoder_model assets/checkpoints/sam/efficientvit_sam_l1_vit_h.encoder.onnx \
    --decoder_model assets/checkpoints/sam/efficientvit_sam_l1_vit_h.decoder.onnx \
    --img_path assets/fig/cat.jpg \
    --out_path assets/demo/onnx_efficientvit_sam_demo.jpg \
    --mode point --point "[[320,240,1]]"
    # --mode boxes --boxes "[150,70,630,400]"

fi