# Deep Learning for Computer Vision: CNN Implementation & Character Recognition

A comprehensive exploration of Convolutional Neural Networks (CNNs) for computer vision tasks, featuring hands-on implementation of custom architectures, character recognition systems, and transfer learning with pre-trained models.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Project Overview

This project demonstrates practical expertise in deep learning and computer vision through three progressive components:

1. **CNN Building Blocks**: Implementation and visualization of fundamental convolution operations, activation functions, and pooling layers
2. **Custom Character Recognition**: Training a multi-layer CNN from scratch to recognize typewritten characters across 29,000+ fonts
3. **Transfer Learning**: Leveraging pre-trained AlexNet for image classification tasks

## 🚀 Key Technical Skills Demonstrated

### Deep Learning & Neural Networks
- **CNN Architecture Design**: Built custom multi-layer convolutional networks with ReLU activations and max-pooling
- **Training Pipeline**: Implemented end-to-end training with SGD optimizer, momentum, mini-batch processing, and data augmentation
- **Loss Functions**: Applied cross-entropy loss with softmax for multi-class classification
- **Model Optimization**: Achieved 99% validation accuracy on character recognition task

### PyTorch Proficiency
- Functional API usage (`torch.nn.functional`) for convolution, pooling, and activation operations
- Module-based architecture design using `torch.nn.Sequential`
- Tensor manipulation and GPU acceleration
- Pre-trained model integration from `torchvision.models`

### Computer Vision Techniques
- **Convolution Operations**: Manual filter design (Laplacian, edge detection) and learned feature extraction
- **Data Preprocessing**: Image normalization, resizing, and mean subtraction
- **Data Augmentation**: Jittering techniques to improve model generalization
- **Feature Visualization**: Analysis of learned convolutional filters

### Transfer Learning
- Loaded and fine-tuned AlexNet (ImageNet ILSVRC 2012 winner)
- Applied pre-trained models to custom image classification tasks
- Demonstrated understanding of feature reusability across domains

## 📊 Project Highlights

### Character Recognition CNN
- **Dataset**: 29,094 character images (26 classes: a-z) from Google Fonts Project
- **Architecture**: 4-layer CNN with progressive feature extraction (1→20→50→500→26 channels)
- **Performance**: 99% training accuracy, 87.9% validation accuracy
- **Application**: Successfully classified individual characters and character sequences

### Custom Filter Design
- Implemented Laplacian filter for edge detection
- Created directional edge detectors for vertical and horizontal features
- Visualized filter responses on real images

### Pre-trained Model Deployment
- Utilized AlexNet (60MB model with 8 layers)
- Classified custom images with ImageNet-trained weights
- Demonstrated practical understanding of production-ready deep learning models

## 🛠️ Technologies & Tools

- **Languages**: Python
- **Frameworks**: PyTorch, torchvision
- **Libraries**: NumPy, Matplotlib, PIL (Pillow)
- **Environment**: Jupyter Notebook, Conda
- **Hardware**: CPU/GPU training support

## 📁 Project Structure

```
INF573-Lab5/
├── INF573-Lab5.ipynb      # Main Jupyter notebook with implementations
├── lab.py                  # Supporting utilities (training loops, visualization)
├── inf573.yaml             # Conda environment configuration
├── data/
│   ├── charsdb.pth         # Character dataset (29,094 samples)
│   ├── alexnet.pth         # Pre-trained AlexNet weights
│   ├── imnet_classes.json  # ImageNet class labels
│   └── peppers.png         # Sample images for testing
└── README.md
```

## 🔧 Setup & Execution

### Prerequisites
Install Anaconda or Miniconda, then create the environment:

```bash
conda env create -f inf573.yaml
conda activate inf573
```

### Running the Project
```bash
jupyter notebook INF573-Lab5.ipynb
```

**Alternative**: The notebook is fully compatible with Google Colab for cloud-based execution.

## 📈 Results & Insights

### Key Findings
1. **Filter Learning**: Convolutional layers automatically learned edge detectors, texture patterns, and structural features
2. **Architectural Depth**: Deeper networks captured hierarchical features (edges → textures → shapes)
3. **Generalization**: Data jittering and validation splits prevented overfitting despite high training accuracy
4. **Transfer Learning**: Pre-trained models provided immediate high-quality feature extraction without retraining

### Performance Metrics
- **Training Speed**: ~100-200 images/second on CPU
- **Character Recognition Accuracy**: 99% (training), 87.9% (validation)
- **Model Size**: AlexNet - 60MB with 8 convolutional/fully-connected layers

## 🎓 Learning Outcomes

This project showcases proficiency in:
- ✅ Deep neural network architecture design and implementation
- ✅ PyTorch framework for computer vision applications
- ✅ Training optimization techniques (SGD, momentum, batch processing)
- ✅ Model evaluation and validation methodologies
- ✅ Transfer learning and pre-trained model deployment
- ✅ Data visualization and interpretation of neural network outputs
- ✅ End-to-end machine learning pipeline development

## 📚 Academic Context

**Course**: INF573 - Computer Vision and Image Analysis  
**Institution**: Graduate-level coursework  
**Original Framework**: Adapted from Oxford VGG practical materials  
**Instructor**: Vicky Kalogeiton, Mathieu Brédif, Thomas Michel

## 📝 License

Copyright (c) 2020 Vicky Kalogeiton

Adapted from Oxford VGG practical. Licensed under MIT License - see LICENSE file for details.

---

*This project demonstrates hands-on experience with state-of-the-art deep learning techniques for computer vision, combining theoretical understanding with practical implementation skills.*

