# exposure_fusion_library
MEF (Multi-Exposure Fusion) README

The provided code is an implementation of Multi-Exposure Fusion (MEF) for improving the quality of photographs captured in environments with varying lighting conditions.
Description

The MEF class provides a method for blending a series of images taken at different exposure levels into a single high-quality image. This is accomplished by leveraging various weighting criteria (contrast, saturation, and well-exposedness) and a Laplacian pyramid.
Setup

    Ensure that you have Python 3.x installed on your machine.

    The code requires the following Python libraries:
        os
        cv2 (OpenCV)
        numpy

    If you haven't already, you can install them using pip:

    pip install opencv-python numpy

How to Use

    Loading Images: Use the static method image_loader to load images from a specified directory. This method filters files based on the provided file extensions and loads each image using OpenCV.

    Example:

    python

image_list = MEF.image_loader("./path_to_images", (".jpg", ".png"))

Processing Images: Use the process method to perform the fusion. This method expects a list of images (in numpy format), a target grayscale mean for brightness correction (gray), and a pixel_balance that determines the exposure target.

Example:

python

result = MEF.process(numpy_image_list=image_list, gray=60, pixel_balance=0.15)

Saving the Result: The processed image can then be saved using OpenCV's imwrite function:

python

    cv2.imwrite("output_filename.bmp", result)

    The if __name__ == "__main__": block at the end of the script provides an example of how to load, process, and save the resultant image.

Key Methods and Their Descriptions

    image_loader(inumpyut_folder, ext): Loads and returns a list of images from the specified folder with the given extensions.

    process(numpy_image_list, gray, pixel_balance): Main function to perform the multi-exposure fusion on the list of images.

    construct_weight_map(numpy_image_list, pixel_balance): Constructs the weight map for the images based on contrast, saturation, and well-exposedness.

    contrast(numpy_image_list): Computes a contrast indicator for the list of images.

    saturation(numpy_image_list): Computes saturation for the list of images.

    well_exposedness(numpy_image_list, pixel_balance): Computes the well-exposedness for the list of images.

    multiresolution_blending(numpy_image_list): Uses a Laplacian pyramid to blend the images at multiple resolutions.

    adjust_brightness(img, target_mean): Adjusts the brightness of the resultant image to a specified mean value.

Note

    Ensure that all the images provided for processing have the same dimensions.
    The function assumes that images are in the format such that their values are in the range [0, 255].
