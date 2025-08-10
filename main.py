# from sam2_extract import *
from utils import *

firstimgpath = 'data/data_2D/can_chowder_000001.jpg'
firstimgmaskpath = 'data/data_2D/can_chowder_000001_1_gt.png'
[xmin,xmax,ymin,ymax] = process_img_png_mask(firstimgpath,firstimgmaskpath,visualize=True)

secondimgpath = 'data/data_2D/can_chowder_000002.jpg'
secondimg = Image.open(secondimgpath)
plt.imshow(secondimg)
plt.show()

op = track_item_boxes(firstimgpath,secondimgpath,[([xmin,xmax,ymin,ymax],1)],visualazi=True)

output_masks = op[1] # Mask for output image is always on op[1] for this example
print(output_masks)

relevant_mask = output_masks[1]
print(relevant_mask)