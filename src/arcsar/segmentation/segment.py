import os
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results

import torch
import torchvision
import numpy as np

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

# Enable TF32 for faster computation on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def segment(processor, image) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        # Load an image
        inference_state = processor.set_image(image)
        # Prompt the model with text
        output = processor.set_text_prompt(state=inference_state, prompt="sky")

    return output["masks"], output["boxes"], output["scores"]


def convert_mask_to_img(masks):
    masks_np = masks.squeeze().cpu().numpy()
    mask_image = (masks_np * 255).astype(np.uint8)
    return Image.fromarray(mask_image)


def overlay_mask_on_img(pil_mask, original_image):
    # Convert PIL mask to numpy boolean array
    mask_np = np.array(pil_mask) > 127  # Threshold to get boolean mask
    # Create RGBA overlay with same dimensions as image
    overlay = np.zeros((*mask_np.shape, 4), dtype=np.uint8)
    overlay[mask_np] = [255, 0, 0, 128]

    # Convert to PIL
    overlay_img = Image.fromarray(overlay, mode="RGBA")
    original_image_rgba = original_image.convert("RGBA")

    # Composite
    result = Image.alpha_composite(original_image_rgba, overlay_img)
    return result


def save_image(output_dir, image) -> None:
    os.makedirs(output_dir, exist_ok=True)
    image.save(f"{output_dir}/out.png")


def main():
    # Load the model
    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    image_name: str = "frame_00001.png"
    image_path: str = f"./repo/assets/images/{image_name}"
    image = Image.open(image_path)

    masks, boxes, scores = segment(processor, image)
    # Convert mask tensor to a PIL Image
    # Masks tensor -> [n_batches, n_masks_on_img, height, width]
    pil_mask = convert_mask_to_img(masks)
    overlay_img = overlay_mask_on_img(pil_mask, image)
    save_image("./output", overlay_img)
    # pil_mask.show()
    # image.show()
    # overlay_img.show()


if __name__ == "__main__":
    main()
