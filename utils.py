import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import os,glob,shutil
import matplotlib.patches as patches

checkpoint = 'sam2_hiera_tiny.pt'
model_cfg = 'sam2_hiera_t.yaml'
# checkpoint = "./sam2_hiera_tiny.pt"
# model_cfg = "sam2_hiera_t.yaml"


predictor_prompt = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
sam2 = build_sam2(model_cfg, checkpoint, device='cuda', apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(sam2)
predictor_vid = build_sam2_video_predictor(model_cfg, checkpoint, device='cuda')
tempfolder = "./tempdir"

def create_if_not_exists(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)

def cleardir(tempfolder):
    filepaths = glob.glob(tempfolder+"/*")
    for filepath in filepaths:
        os.unlink(filepath)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size)
           
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), linewidth=2))

def process_img_png_mask(img_path, mask_path, visualize=False):
    """Compute [xmin, xmax, ymin, ymax] from a binary PNG mask."""
    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    arr = np.array(mask) > 0
    ys, xs = np.where(arr)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError(f"No positive pixels in mask: {mask_path}")
    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())
    if visualize:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(img)
        show_box([xmin, ymin, xmax, ymax], ax)
        ax.set_title("Ground-truth box from mask")
        create_if_not_exists("./outputs")
        plt.savefig("./outputs/gt_box_from_mask.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    # Return in same order you used originally: [xmin,xmax,ymin,ymax]
    return [xmin, xmax, ymin, ymax]

def track_item_boxes(imgpath1,imgpath2,img1boxclasslist,visualize=True):
    # imgpath1 :: Image where object is known
    # imgpath2 :: Image where object is to be tracked
    # img1boxclasslist :: [ ([xmin,xmax,ymin,ymax],objectnumint) ,....] for all objects in imagepath1
    create_if_not_exists(tempfolder)
    cleardir(tempfolder)
    shutil.copy(imgpath1,tempfolder+"/00000.jpg")
    shutil.copy(imgpath2,tempfolder+"/00001.jpg")

    inference_state = predictor_vid.init_state(video_path="./tempdir")
    predictor_vid.reset_state(inference_state)

    ann_frame_idx = 0
    for img1boxclass in img1boxclasslist:
        ([xmin,xmax,ymin,ymax],objectnumint) = img1boxclass
        box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
        _, out_obj_ids, out_mask_logits = predictor_vid.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=objectnumint,
            box=box,
        )
    video_segments = {} # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor_vid.propagate_in_video(
        inference_state=inference_state
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    if visualize:
        fig, ax = plt.subplots()
        plt.title(f"original image object ::")
        ax.imshow(Image.open(tempfolder+"/00000.jpg"))
        rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, 
                                 edgecolor = 'red', facecolor='none')
        ax.add_patch(rect)
        plt.show()
        out_frame_idx = 1
        plt.figure(figsize=(6, 4))
        plt.title(f"detected object in test image ::")
        plt.imshow(Image.open(tempfolder+"/00001.jpg"))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
    return video_segments