# ASL Sign Language Translation

This project builds a translation engine that recognizes an American Sign Language hand sign from a static image and predicts its letter.

The project compares two approaches:

1. **Control:** a classical machine-learning model, starting with an RBF Support Vector Machine (SVM).
2. **Experiment:** a multilayer perceptron (MLP) neural network built with PyTorch or TensorFlow.

The best-performing model will be packaged as a local application that accepts a PNG upload and displays the predicted letter and confidence score.

> A clean, reproducible, versioned, and deployed pipeline is the goal. Accuracy is only one part of the evaluation.

## Dataset

The project uses the [Sign Language MNIST dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist).

The dataset contains 28x28 grayscale images and separate training and test CSV files:

- `sign_mnist_train.csv`
- `sign_mnist_test.csv`

The dataset's labels represent the static ASL alphabet classes. `J` and `Z` are not included because they require motion.

## Project Structure

```text
.
├── notebooks/
│   └── projectSVM.ipynb       # Week 1 classical-model notebook
├── Models/
│   └── asl_svm_model.pkl       # Exported SVM model
└── README.md                  # Project documentation
```

Open the baseline notebook with:

```bash
jupyter lab notebooks/projectSVM.ipynb
```