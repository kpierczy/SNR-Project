# Run 1

1. Epochs: `60`
2. Batch size: `64`
3. Kernel initializer: `zeros`
4. Bias initializer: `zeros`
5. Optimiser: `adam`
6. Loss function: `categorical crossentropy`
7. Learinng rate: `adaptive` (ReduceLROnPlateau):
    - initial: 1e-3
    - factor: 1e-1
    - minimal: 1e-6
    - minimal delta: 1e-6
    - patience: 8
    - cooldown: 0
8. Test type: last model
9. Validation:Test ratio: 1:1

# Run 2

1. Epochs: `40`
2. Batch size: `64`
3. Kernel initializer: `zeros`
4. Bias initializer: `zeros`
5. Optimiser: `adam`
6. Loss function: `categorical crossentropy`
7. Learinng rate: `adaptive` (ReduceLROnPlateau):
    - initial: 1e-3
    - factor: 1e-1
    - minimal: 1e-7
    - minimal delta: 1e-7
    - patience: 4
    - cooldown: 0
8. Test type: best model
9. Validation:Test ratio: 1:1
