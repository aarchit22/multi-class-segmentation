# multi-class-segmentation

The method of using this is to first run the dependencies.ipynb file to download all the requirements, then use this command:
python segmentation.py -i 'input_directory_to_rgb_image' --target_class 'person' --show --conf_threshold 0.95

Here,
segmentation.py is our file which contains the models used in trimap, alpha matte and foreground estimation.

This model shows excellent results for humans, cats, dogs, horses, among 20 other object classes.
