import os
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

# Enable TF32 for faster computation on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def segment(processor, image):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        # Load an image
        inference_state = processor.set_image(image)
        # Prompt the model with text
        output = processor.set_text_prompt(state=inference_state, prompt="sky")

    return output["masks"], output["boxes"], output["scores"]


def convert_mask_to_img(masks):
    """
    Moves tensor mask from GPU to CPU and then converts it to a numpy array (boolean).
    Removes batch dimension and takes the union of multiple masks exist.
    Then to a grayscale array and then to a PIL image.
    """
    # Masks tensor -> [n_batches, n_masks_on_img, height, width]
    masks_np = masks.cpu().numpy()

    # Remove batch dimension if present
    if masks_np.ndim == 4:
        masks_np = masks_np[0]

    # Combine multiple masks into one by taking the max (union of masks)
    if masks_np.ndim == 3:
        combined_mask = masks_np.max(axis=0)  # Shape: [height, width]
    else:
        combined_mask = masks_np
    mask_image = (combined_mask * 255).astype(np.uint8)
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


def load_images(folder_path: str, extensions: tuple = (".jpg", ".jpeg")) -> list[Path]:
    """
    Load all image paths from a folder.
    """
    folder = Path(folder_path)
    image_paths = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in extensions]
    )
    return image_paths


def save_image(output_dir: str, name: str, image) -> None:
    os.makedirs(output_dir, exist_ok=True)
    image.save(f"{output_dir}/{name}_masked.png")


def main():
    # Load the model
    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    # image_name: str = "frame_00001.png"
    # image_path: str = f"./repo/assets/images/{image_name}"
    # image = Image.open(image_path)
    input_dir: str = "./output/video"
    image_paths = load_images(input_dir)
    print(f"Found {len(image_paths)} images to mask")

    for i, image_path in enumerate(image_paths):
        print(f"Processing [{i + 1}/{len(image_paths)}]: {image_path.name}")
        image = Image.open(image_path)
        masks, boxes, scores = segment(processor, image)
        pil_mask = convert_mask_to_img(masks)
        overlay_img = overlay_mask_on_img(pil_mask, image)
        save_image("./output/masked", image_path.name, overlay_img)


if __name__ == "__main__":
    main()
