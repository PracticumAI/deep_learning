# PyTorch Conversion of Deep Learning Notebooks

This repository has been converted from TensorFlow/Keras to PyTorch and PyTorch Lightning. The converted notebooks provide the same learning experience while leveraging the PyTorch ecosystem.

## What Changed

### Main Conversions:
1. **02.1_code_a_perceptron.ipynb**: Converted from TensorFlow to pure PyTorch
2. **02.2_mnist_classifier.ipynb**: Converted from TensorFlow/Keras to PyTorch Lightning
3. **03_bees_vs_wasps.ipynb**: Converted from TensorFlow/Keras to PyTorch Lightning

### Key Technical Changes:

#### From TensorFlow to PyTorch:
- `tf.Variable` → `torch.tensor` with `requires_grad=True`
- `tf.keras.Sequential` → `pl.LightningModule` classes
- `tf.optimizers.SGD` → `torch.optim.SGD`
- `model.fit()` → `trainer.fit()`
- `model.evaluate()` → Manual evaluation loops

#### New Dependencies:
- `torch` - Core PyTorch library
- `torchvision` - Computer vision utilities
- `pytorch-lightning` - High-level PyTorch framework
- `torchmetrics` - Metrics for PyTorch models

## Environment Setup

### Option 1: Using pip (requirements.txt)
```bash
pip install -r requirements.txt
```

### Option 2: Using conda (environment.yml)
```bash
conda env create -f environment.yml
conda activate practicum_pytorch
```

### Option 3: Manual Installation
```bash
# Core PyTorch (adjust CUDA version as needed)
pip install torch torchvision pytorch-lightning

# Additional dependencies
pip install pandas numpy matplotlib scikit-learn jupyter
```

## Key Differences for Students

### 1. Model Definition
**Before (TensorFlow/Keras):**
```python
model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(50, activation='relu'),
    Dense(10, activation='softmax')
])
```

**After (PyTorch Lightning):**
```python
class MNISTClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x
```

### 2. Training
**Before:**
```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(train_data, epochs=10)
```

**After:**
```python
trainer = Trainer(max_epochs=10)
trainer.fit(model, train_loader)
```

### 3. Data Loading
**Before (TensorFlow):**
```python
(train_features, train_labels), (test_features, test_labels) = tf.keras.datasets.mnist.load_data()
```

**After (PyTorch):**
```python
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

## Benefits of PyTorch Lightning

1. **Cleaner Code**: Separates research code from engineering code
2. **Built-in Best Practices**: Automatic logging, checkpointing, and distributed training
3. **Flexibility**: Easy to extend and customize
4. **Industry Standard**: Widely used in research and production

## Running the Notebooks

1. Ensure you have the `practicum_pytorch` kernel selected
2. Install the required packages using one of the methods above
3. Run the notebooks in order:
   - `02.1_code_a_perceptron.ipynb` - Basic perceptron from scratch
   - `02.2_mnist_classifier.ipynb` - MNIST digit classification
   - `03_bees_vs_wasps.ipynb` - Image classification with hyperparameter tuning

## Notes for Instructors

- The learning objectives remain the same
- Students will gain exposure to PyTorch, which is increasingly popular in research
- PyTorch Lightning provides structure while maintaining PyTorch's flexibility
- Code is more explicit about training loops and model components
- Better preparation for advanced deep learning concepts

## Troubleshooting

### Common Issues:
1. **CUDA compatibility**: Ensure PyTorch CUDA version matches your system
2. **Memory issues**: Reduce batch size if encountering out-of-memory errors
3. **Data loading**: Ensure data paths are correct (unchanged from original notebooks)

### Performance Notes:
- Training times should be similar to TensorFlow versions
- PyTorch Lightning adds minimal overhead
- GPU utilization should be comparable

## Additional Resources

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Lightning Documentation](https://pytorch-lightning.readthedocs.io/)
- [PyTorch vs TensorFlow Comparison](https://pytorch.org/tutorials/beginner/tensor_tutorial.html)
