import os
import cv2
import numpy

from typing import Union
from pathlib import Path


class MEF:
    def __init__(self):
        # Declare values assigned later
        self.R = None
        self.pyr = None
        self.well_exposedness_stack = None
        self.saturation_stack = None
        self.contrast_stack = None
        self.well_exposedness_param = None
        self.saturation_param = None
        self.contrast_param = None
        self.width = None
        self.height = None
        self.N = None
        self.W = None

        self.weight_param = numpy.array([1, 1, 1], dtype=numpy.float32)
        self.nlev = 12
        # Exposure target. If overexposed regions should be avoided - lover value and vice versa
        self.exposure_target = 0.5

    # Function for direct load of images
    @staticmethod
    def image_loader(inumpyut_folder: str, ext: Union[str, tuple]) -> list:
        """
        Loads images from a specified folder with a specified extension(s).

        This function lists all the files in the given folder, filters them based on the
        provided file extension(s), and loads each image using OpenCV. The loaded images
        are converted to numpy float32 and normalized to a range between 0 and 1.

        Args:
        - inumpyut_folder (str): The path to the folder containing images.
        - ext (Union[str, tuple]): The extension or tuple of extensions of images to be loaded.
                                   Example: ('.jpg', '.png')

        Returns:
        - list: A list of loaded images in cv2 (numpy) format.

        Nested Function:
        - common_prefix(strs: list) -> str:
            Computes the longest common prefix string amongst an array of strings.
            Args:
            - strs (list): List of strings to determine the common prefix.

            Returns:
            - str: The longest common prefix, or empty string if no common prefix exists.

        Note:
        The function assumes that OpenCV (cv2) and numpy have been imported.
        """
        def common_prefix(strs: list) -> str:
            """
            Compute the longest common prefix string amongst an array of strings.

            Given a list of strings, this function determines the longest common prefix
            shared amongst all elements in the list. If no common prefix exists, the
            function returns an empty string.

            Args:
            - strs (list): List of strings to determine the common prefix.

            Returns:
            - str: The longest common prefix, or empty string if no common prefix exists.

            Example:
            '>>> common_prefix(["flower","flow","flight"])'
            'fl'
            '>>> common_prefix(["dog","car","race"])'
            ''
            """
            if not strs:
                return ""
            prefix = strs[0]
            for i in range(1, len(strs)):
                while strs[i].find(prefix) != 0:
                    prefix = prefix[:-1]
                    if not prefix:
                        return ""
            return prefix
        # Get list of files in folder
        inumpyut_content = os.listdir(inumpyut_folder)

        # Filter out by extensions
        image_names = [file for file in inumpyut_content if file.endswith(ext)]

        # Extract common part of filenames
        common_name = common_prefix(image_names)

        # Declare list for loaded cv2 images
        images = []

        # Fill list of cv2 images
        for image_name in image_names:
            image_path = str(Path(inumpyut_folder) / image_name)
            cv2_image = cv2.imread(image_path).astype(numpy.float32) / 255.0
            images.append(cv2_image)

        # Return list of cv2 images
        return images

    ### Main function ###

    @classmethod
    def process(cls, numpy_image_list: list, gray: int = 60, pixel_balance: float = 0.5):
        instance = cls()  # Create an instance of MEF
        instance.construct_weight_map(numpy_image_list=numpy_image_list, pixel_balance=pixel_balance)
        instance.multiresolution_blending(numpy_image_list=numpy_image_list)
        instance.R = instance.pyr[instance.nlev - 1].copy()

        for l in range(instance.nlev - 2, -1, -1):
            GE = cv2.pyrUp(instance.R)
            temp = cv2.resize(GE, (instance.pyr[l].shape[:2][1], instance.pyr[l].shape[:2][0]))
            instance.R = instance.pyr[l] + temp

        fused_image = instance.R * 255
        bright_corrected_image = instance.adjust_brightness(fused_image, target_mean=gray)

        return bright_corrected_image

    ### Three-component weight map construction block ###

    def construct_weight_map(self, numpy_image_list: list, pixel_balance: float = 0.5):
        # Get base balance parameters
        self.contrast_param, self.saturation_param, self.well_exposedness_param = self.weight_param

        # Get image set characteristics TODO: Check for image with non same sizes
        self.N = len(numpy_image_list)  # the number of images
        self.width = len(numpy_image_list[0])
        self.height = len(numpy_image_list[0][0])

        # If the contrast parameter is positive, compute the contrast for the given images
        # and raise the result to the power of contrast_param.
        if self.contrast_param > 0:
            self.contrast_stack = numpy.power(self.contrast(numpy_image_list), self.contrast_param)

        # Similarly, if the saturation parameter is positive, compute the saturation for the given images
        # and raise the result to the power of saturation_param.
        if self.saturation_param > 0:
            self.saturation_stack = numpy.power(self.saturation(numpy_image_list), self.saturation_param)

        # Lastly, if the well_exposedness parameter is positive, compute the well-exposedness for the given images
        # and raise the result to the power of well_exposedness_param.
        if self.well_exposedness_param > 0:
            self.well_exposedness_stack = numpy.power(self.well_exposedness(numpy_image_list,
                                                                            pixel_balance=pixel_balance),
                                                      self.well_exposedness_param)

        # Calculate the final weight W by multiplying the individual metrics: contrast_stack, saturation_stack,
        # and well_exposedness_stack.
        self.W = numpy.multiply(self.contrast_stack,
                                numpy.multiply(self.saturation_stack, self.well_exposedness_stack))

        # Unification and normalization of image stack
        W_sum = self.W.sum(axis=0)
        self.W = numpy.divide(self.W, W_sum + 1e-12)
        numpy.seterr(divide='ignore', invalid='ignore')

    """
    Apply a Laplacian filter to the grayscale version of each image, and take the absolute value ofthe filter response.
    This yields a simple indicator C for contrast. It tends to assign a high weight to important elements such as edges
    and texture. A similar measure was used for multi-focus fusion for extended depth-of-field.
    """
    def contrast(self, numpy_image_list: list):
        contrast = numpy.zeros((self.N, self.width, self.height), dtype=numpy.float32)
        kernel = numpy.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=numpy.float32)
        for i in range(self.N):
            mono = cv2.cvtColor(numpy_image_list[i], cv2.COLOR_RGB2GRAY)
            contrast[i] = abs(cv2.filter2D(mono, -1, kernel))
        return contrast

    """
    As a photograph undergoes a longer exposure, the resulting colors become desaturated and eventually clipped.
    Saturated colors are desirable and make the image look vivid. We include a saturation measure S, which is computed
    as the standard deviation within the R, G and B channel, at each pixel.
    """
    def saturation(self, numpy_image_list: list):
        saturation = numpy.zeros((self.N, self.width, self.height), dtype=numpy.float32)
        for i in range(self.N):
            # saturation is computed as the standard deviation of the color channels
            B, G, R = cv2.split(numpy_image_list[i])
            saturation[i] = numpy.std([B, G, R], axis=0) * 256
        return saturation

    """
    Looking at just the raw intensities within a channel, reveals how well a pixel is exposed.
    We want to keep intensities that are not near zero (underexposed) or one (overexposed). We weight each intensity
    i based on how close it is to 0.5 using a Gauss curve: exp(−(i−0.5)^2/2σ^2), where σ equals 0.2 in current
    implementation. To account for multiple color channels, we apply the Gauss curve to each channel separately,
    and multiply the results, yielding the measure E
    """
    def well_exposedness(self, numpy_image_list: list, pixel_balance: float = 0.5):
        sig = .2
        sig2 = sig * sig
        exposedness = numpy.zeros((self.N, self.width, self.height), dtype=numpy.float32)
        for i in range(self.N):
            b, g, r = cv2.split(numpy_image_list[i])
            R = numpy.exp(-(numpy.power(r - pixel_balance, 2) / (2 * sig2)))
            G = numpy.exp(-(numpy.power(g - pixel_balance, 2) / (2 * sig2)))
            B = numpy.exp(-(numpy.power(b - pixel_balance, 2) / (2 * sig2)))

            exposedness[i] = numpy.multiply(R, numpy.multiply(G, B))
        return exposedness

    ### Laplacian pyramid construction block ###

    def multiresolution_blending(self, numpy_image_list: list):
        # this function is to get the Laplacian pyramid eq.3

        # initialize image pyramid for reconstruction
        w, h = self.width, self.height
        self.pyr = [numpy.zeros((w, h, 3), dtype=numpy.float32)]
        for _ in range(self.nlev):
            w = int(numpy.ceil(w / 2))
            h = int(numpy.ceil(h / 2))
            self.pyr.append(numpy.zeros((w, h, 3), dtype=numpy.float32))

        # construct the image pyramid
        for i in range(self.N):
            pyrW = self.gaussianumpyyrWeightMap(self.W[i])
            pyrI = self.laplacianumpyyrImage(numpy_image_list[i])

            for l in range(self.nlev):
                r, c = len(pyrW[l]), len(pyrW[l][0])
                b = numpy.zeros((r, c, 3), dtype=numpy.float32)

                b[:, :, 0] = pyrW[l]
                b[:, :, 1] = pyrW[l]
                b[:, :, 2] = pyrW[l]

                self.pyr[l] += numpy.multiply(b, pyrI[l])
                
    def gaussianumpyyrWeightMap(self, W_k):
        WC = W_k.copy()
        pyrW = [WC]
        for i in range(1, self.nlev):
            WC = cv2.pyrDown(cv2.GaussianBlur(WC, (3, 3), 0))
            pyrW.append(WC)

        return pyrW

    def laplacianumpyyrImage(self, I):
        pyrL = []

        J = I.copy()
        for i in range(0, self.nlev - 1):
            src = cv2.GaussianBlur(J, (3, 3), 0)
            I = cv2.pyrDown(src)
            GAU = cv2.pyrUp(I)
            temp = cv2.resize(GAU, (J.shape[:2][1], J.shape[:2][0]))
            L = cv2.subtract(J, temp)

            pyrL.append(L)
            J = I.copy()

        pyrL.append(I)
        return pyrL
    
    ### Brightness correction ###

    def adjust_brightness(self, img, target_mean=60):
        """
        Adjust the brightness of an image to adwda specific mean value.

        Parameters:
        - img: the inumpyut image (BGR format)
        - target_mean: the desired mean brightness value

        Returns:
        - adjusted_img: the brightness-adjusted image
        """

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calculate the difference to adjust brightness
        brightness_diff = target_mean - numpy.mean(gray)

        # Convert image to float32 to avoid potential underflow/overflow issues
        float_image = img.astype(numpy.float32)
        adjusted_image = numpy.clip(float_image + brightness_diff, 0, 255).astype(numpy.uint8)

        return adjusted_image


if __name__ == "__main__":
    image_list = MEF.image_loader("./imtest", (".png", ".bmp"))
    result = MEF.process(numpy_image_list=image_list, gray=60, pixel_balance=0.15)
    cv2.imwrite("1.bmp", result)
