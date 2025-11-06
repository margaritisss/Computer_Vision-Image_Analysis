# Stereo Vision and Epipolar Geometry

A comprehensive implementation of fundamental computer vision techniques for stereo image analysis, demonstrating advanced understanding of 3D reconstruction, camera geometry, and feature matching algorithms.

## 🎯 Overview

This lab implements stereo vision algorithms to analyze image pairs taken from different viewpoints. The implementation covers the complete pipeline from camera calibration to feature correspondence, showcasing practical applications in 3D scene reconstruction and depth estimation.

## 🔑 Key Technical Concepts

### 1. **Camera Projection & Geometry**
- Implemented camera projection matrices for 3D-to-2D point transformations
- Computed camera centers using homogeneous coordinates and matrix inversion
- Applied linear algebra techniques (matrix multiplication, inverse operations)

### 2. **Epipolar Geometry**
- Calculated epipoles (projections of camera centers in opposite images)
- Derived fundamental matrices relating corresponding points between stereo images
- Implemented epipolar line computation and visualization

### 3. **Stereo Correspondence Matching**
- Developed Zero-Mean Normalized Cross-Correlation (ZNCC) algorithm
- Applied window-based template matching along epipolar lines
- Implemented robust matching with brightness invariance

## 💻 Technical Skills Demonstrated

- **Python Programming**: NumPy array operations, object-oriented design
- **Computer Vision**: OpenCV, scikit-image
- **Linear Algebra**: Matrix operations, homogeneous coordinates, projective geometry
- **Image Processing**: Patch extraction, correlation metrics, feature matching
- **Data Visualization**: Matplotlib, interactive plotting with callbacks
- **Mathematical Modeling**: 3D geometry, camera models, epipolar constraints

## 🛠️ Technologies Used

```python
- Python 3.x
- NumPy (numerical computing)
- OpenCV (computer vision)
- scikit-image (image I/O)
- Matplotlib (visualization)
- ipywidgets (interactive controls)
```

## 📊 Implementation Highlights

### Camera Model
```python
class Camera:
    - Projection matrix representation (3×4)
    - Center point computation
    - 3D to 2D point projection
```

### Core Algorithms
1. **Epipole Calculation**: Projects camera center into opposite image space
2. **Fundamental Matrix**: Relates corresponding points via epipolar geometry
3. **ZNCC Matching**: Finds correspondences with normalized cross-correlation

### Interactive Visualization
- Click on any point in either image to:
  - Display corresponding epipolar line in the opposite image
  - Visualize ZNCC scores along the epipolar line
  - Automatically identify best matching point

## 🎓 Academic Context

**Course**: Computer Vision and Image Analysis  
**Lab**: Stereo Imagery (Lab 4)  
**Author**: Georgios Evangelos Margaritis

## 🚀 Applications

This implementation demonstrates foundational techniques used in:
- 3D reconstruction from stereo images
- Depth estimation for autonomous vehicles
- Structure from Motion (SfM)
- Augmented Reality (AR) systems
- Robotics and visual navigation

## 📈 Results

The implementation successfully:
- ✅ Computes accurate camera parameters and epipoles
- ✅ Generates precise epipolar lines for feature correspondence
- ✅ Matches corresponding points with high accuracy using ZNCC
- ✅ Provides interactive visualization for validation

## 🔍 Key Insights

- **Epipolar Constraint**: Reduces 2D search space to 1D (along epipolar line)
- **ZNCC Robustness**: Zero-mean normalization provides illumination invariance
- **Geometric Validation**: Epipolar geometry ensures geometrically consistent matches

## 📝 Usage

```python
# Load stereo image pair
image1 = io.imread('face00.tif')
image2 = io.imread('face01.tif')

# Initialize cameras with calibration parameters
cam1 = Camera([...])
cam2 = Camera([...])

# Compute fundamental matrix
F12 = fundamental(cam1, cam2)

# Interactive matching visualization
onclick_draw_NCC_match_and_epipolar_line(axeslist, [0,0], (302,196))
```

## 🎯 Skills for Recruiters

This project demonstrates proficiency in:
- **Mathematical Foundations**: Linear algebra, projective geometry, optimization
- **Algorithm Implementation**: From theory to working code
- **Software Engineering**: Clean code, modular design, documentation
- **Problem Solving**: Complex multi-step computer vision pipeline
- **Practical CV Applications**: Real-world stereo vision techniques

---

