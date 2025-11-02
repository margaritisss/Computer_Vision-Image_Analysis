# Image Processing and Computer Vision Fundamentals

## Overview

This lab serves as a practical demonstration of fundamental image processing and computer vision techniques. Through a series of hands-on exercises in a Jupyter Notebook, this lab explores the manipulation of digital images, color spaces, and the implementation of basic image filtering. The goal is to showcase a strong understanding of how images are represented and can be programmatically altered using popular Python libraries.

This repository is an excellent showcase of my ability to work with image data, implement core computer vision algorithms, and present technical work in a clear and organized manner.

## Key Features & Concepts Demonstrated

*   **Image I/O and Display:** Loading, displaying, and handling image data using `scikit-image` and `matplotlib`.
*   **Color Space Manipulation:**
    *   **RGB & BGR:** Deep dive into the structure of color images, including extracting and visualizing individual Red, Green, and Blue channels.
    *   **RGB to BGR Conversion:** Implemented color channel swapping both with OpenCV and from scratch using NumPy array manipulation.
    *   **Grayscale Conversion:** Demonstrated multiple techniques for converting color images to grayscale, including simple averaging and a weighted luminosity method based on human perception.
    *   **HSV (Hue, Saturation, Value):** Explored the HSV color space and its components, including an interactive visualization of how changes in H, S, and V affect an image.
*   **Geometric Transformations:** Performed basic image transformations such as horizontal and vertical flipping.
*   **Image Filtering & Analysis:**
    *   **Image Gradients:** Calculated image gradients using both OpenCV's built-in Sobel filter and a custom implementation of a finite difference scheme to detect edges and textures.
*   **Python for Data Science:**
    *   **NumPy:** Utilized for efficient and powerful n-dimensional array manipulation, which is at the core of image processing.
    *   **OpenCV:** Leveraged for industry-standard computer vision functions.
    *   **Matplotlib:** Used for creating clear and informative visualizations of images and their components.
    *   **Jupyter Notebook & ipywidgets:** Created an interactive and well-documented environment for experimentation and analysis.

## Libraries Used

*   `numpy`
*   `opencv-python`
*   `scikit-image`
*   `matplotlib`
*   `ipywidgets`

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install numpy opencv-python scikit-image matplotlib ipywidgets
    ```
3.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
4.  Open the `lab_1.ipynb` file and run the cells to see the image manipulations and visualizations.

