# Computer Vision: Canny Edge and Harris Corner Detection from Scratch

This lab is a Jupyter Notebook that provides a detailed, hands-on implementation of fundamental computer vision algorithms for feature extraction. It demonstrates a strong understanding of the mathematical principles and practical coding skills required to build classic computer vision pipelines from the ground up.

The notebook focuses on implementing the Canny edge detector and the Harris corner detector, two of the most important techniques in image analysis, using a sample road image relevant to autonomous driving applications.

## Key Concepts & Implementations

This project showcases the ability to:

*   **Implement Image Gradient Operators:**
    *   **Finite Difference Method:** Wrote a function to compute image gradients using a central difference scheme.
    *   **Sobel Operator:** Implemented the 3x3 Sobel filter from scratch to calculate robust image gradients, demonstrating an understanding of convolution and kernel operations.

*   **Build a Canny Edge Detector from Scratch:**
    *   **Gradient Magnitude & Orientation:** Calculated the magnitude and direction of gradients for every pixel.
    *   **Non-Maximum Suppression:** Implemented the logic to thin edges by suppressing pixels that are not local maxima in the direction of the gradient. This shows a deep understanding of the algorithm's core principles.
    *   **Hysteresis Thresholding:** Developed a queue-based (Breadth-First Search) algorithm to connect strong edges with weak edges, demonstrating the ability to implement graph-like traversal on images.

*   **Implement the Harris Corner Detector from Scratch:**
    *   **Structure Tensor:** Computed the structure tensor matrix (Ixx, Iyy, Ixy) from image gradients.
    *   **Corner Response Function (R):** Implemented the Harris response function `R = det(M) - k * trace(M)^2` to identify corners.
    *   **Thresholding & Visualization:** Applied a threshold to the response function to locate and display the final corners on the original image.

*   **Analyze and Interpret Results:**
    *   Provided clear explanations for why image gradients differ in the X and Y directions.
    *   Articulated the critical trade-offs between high and low thresholds in edge detection.
    *   Tuned algorithm parameters (`low` and `high` thresholds) to achieve optimal results for a specific application (driving scene analysis).

## Technologies Used

*   **Python**
*   **Jupyter Notebook**
*   **NumPy:** For efficient numerical computation and array manipulation.
*   **OpenCV (`cv2`):** Used for baseline comparisons and utility functions like Gaussian blur and color conversion.
*   **Matplotlib & `ipywidgets`:** For data visualization and creating interactive sliders to demonstrate the effect of different parameters.
*   **Scikit-image:** For image loading.

## How to View This Project

1.  Ensure you have a Python environment with Jupyter and the libraries listed above (`numpy`, `opencv-python`, `matplotlib`, `ipywidgets`, `scikit-image`).
2.  Open the `Georgios Evangelos Margaritis.ipynb` file in Jupyter Notebook or Jupyter Lab.
3.  The notebook is self-contained with explanations, code, and output for each step. You can "Run All" cells to see the entire pipeline execute.

