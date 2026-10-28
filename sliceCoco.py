from sahi.slicing import slice_coco

# Slices the original trap pictures and their COCO annotations into partially
# overlapping 640x640 tiles, used to build the training dataset.
# Annotations cut by a slice border are kept if at least min_area_ratio of their area
# falls inside the slice. Slices without insects are kept too (ignore_negative_samples=False).

#Setup your project variables:
# PLEASE EDIT THESE:

coco_dict, coco_path = slice_coco(
    coco_annotation_file_path='/path/to/repo/dataset/Traps.json',
    image_dir='/path/to/repo/dataset/Images/',
    output_coco_annotation_file_name="sliced_Traps.json",
    ignore_negative_samples=False,
    output_dir="/path/to/main/folder/slicedCOCOdataset/",
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    min_area_ratio=0.1,
    verbose=True)
