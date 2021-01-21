# Run 1

1. Epochs: `40`
2. Batch size: `64`
3. Kernel initializer: `glorot-normal`
4. Bias initializer: `glorot-normal`
5. Optimiser: `adam`
6. Loss function: `categorical crossentropy`
7. Learinng rate: `adaptive` (ReduceLROnPlateau):
    - initial: 1e-4
    - factor: 2e-1
    - minimal: 1e-7
    - minimal delta: 5e-2
    - patience: 4
    - cooldown: 0
8. Test type: best model
9. Validation:Test ratio: 1:1
10. Augmentation:
    - vertical flips
    - horizontal flips
    - random rotations in range [-20,20] degrees

# Run 2

1. Epochs: `40`
2. Batch size: `64`
3. Kernel initializer: `glorot-normal`
4. Bias initializer: `glorot-normal`
5. Optimiser: `adam`
6. Loss function: `categorical crossentropy`
7. Learinng rate: `adaptive` (ReduceLROnPlateau):
    - initial: 1e-4
    - factor: 2e-1
    - minimal: 1e-7
    - minimal delta: 5e-2
    - patience: 4
    - cooldown: 0
8. Test type: best model
9. Validation:Test ratio: 1:1
10. Augmentation:
    - vertical flips
    - horizontal flips
    - random rotations in range [-20,20] degrees
    - random brightness correction in range [-25,25]
    - random contrast correction in range [0.9,1.1]
    - random shifts in x direction in range [-25,25]   
    - random shifts in y direction in range [-25,25]    