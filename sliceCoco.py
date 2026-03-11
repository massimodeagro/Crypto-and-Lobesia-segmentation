from sahi.slicing import slice_coco

coco_dict, coco_path = slice_coco(
    coco_annotation_file_path='path/to/coco/dataset/Traps.json',
    image_dir='path/to/coco/dataset/Images/',
    output_coco_annotation_file_name="sliced_Traps.json",
    ignore_negative_samples=False,
    output_dir="path/to/main/folder/slicedCOCOdataset/",
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    min_area_ratio=0.1,
    verbose=True)
