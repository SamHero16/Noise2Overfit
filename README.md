# Noise2Overfit
### Framework for stack data to create early stopping criteria, performance metrics and visualization of Noise2Noise output.

Step 1:
Overfit a Noise2Noise model: be sure to shuffle noisy input and target pairs within stack every epoch (prevents model from prediting noisy target)
- Use Noise2Overfit.py script
- See example on WandB here: https://wandb.ai/samhero/Noise2Overfit/overview?nw=nwusersamhero
