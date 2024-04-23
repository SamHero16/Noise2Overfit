# Noise2Overfit
### Framework for stack data to create early stopping criteria, performance metrics and visualization of Noise2Noise output.

Step 1:
Overfit a Noise2Noise model: be sure to shuffle noisy input and target pairs within stack every epoch (prevents model from prediting noisy target)
- Use [Noise2Overfit.py script](Noise2Overfit.py)
- See example on WandB here: https://wandb.ai/samhero/Noise2Overfit/overview?nw=nwusersamhero
- see [overfit_montage.pdf](overfit_montage.pdf)

Step 2:
Set aside a portion of your images for val and test. I am doing this manually for now.
- I removed 4 tif stacks from my training data: 2 for validation and 2 for test
- Set these aside in another folder in my case [noisyTif-Validation](2023-08-30-noisyTif-Validation) and noisyTest (not uploaded yet)
- Get the corresponding pseudo-clean images, in my case [pseudoClean-Validation](2023-08-30-pseudoClean-Validation) and pseudoCleanTest (not uploaded yet)

Step 3:
Train a new Noise2Noise model.
- Use [Noise2EarlyStop.py script](Noise2EarlyStop.py)
- Use the validation set for early stopping -> stop when the model predicts images closest to the pseudo-clean. MSE(pseudo-clean , model(noisyValidation)

Step 4:
Test network on test set. See how well it removes the noise by comparing model output with pseudo-clean images.

