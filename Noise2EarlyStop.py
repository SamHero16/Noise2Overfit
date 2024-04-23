from PIL import Image
import numpy as np
import os
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import wandb
import pickle

# Torch dataset for batching and shuffling
class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def loadOne2OneTrain(path):
    X = [] # noisy input
    Y = [] # noisy targets
    unique = [] # one image from each stack for easy denoising
    averages = [] # average of stack
    names = [] # names of tif stack

    # loop through folder containing tifs
    for image_class in os.listdir(path):
        currAve = np.zeros([128, 128])
        sameBaseImages = []
        image_path = os.path.join(path, image_class)
        tif = Image.open(image_path)

        # loads images and adds them to list
        for i in range(tif.n_frames):
            tif.seek(i)
            img = np.array(tif)
            sameBaseImages.append(img)
            currAve += img
        averages.append(currAve / tif.n_frames)
        # Shuffle the stack matchups (every epoch)
        shuffled_images = sameBaseImages.copy()
        random.shuffle(shuffled_images)
        X.append(shuffled_images)
        Y.append(sameBaseImages)

        unique.append(shuffled_images[0])
        names.append(image_class)

    # flatten X
    X = np.array([item for row in X for item in row])
    Y = np.array([item for row in Y for item in row])

    averages = np.array(averages)
    unique = np.array(unique)

    unique = torch.from_numpy(unique)
    unique = unique.unsqueeze(1)

    # Image transformations every epoch
    for i in range(X.shape[0]):
        # random rotation
        r = random.random()
        if r <= .25:
            X[i] = np.rot90(X[i])
            Y[i] = np.rot90(Y[i])
        elif .25 < r <= .50:
            X[i] = np.rot90(X[i])
            X[i] = np.rot90(X[i])
            Y[i] = np.rot90(Y[i])
            Y[i] = np.rot90(Y[i])
        elif .50 < r <= .75:
            X[i] = np.rot90(X[i])
            X[i] = np.rot90(X[i])
            X[i] = np.rot90(X[i])
            Y[i] = np.rot90(Y[i])
            Y[i] = np.rot90(Y[i])
            Y[i] = np.rot90(Y[i])
        # random flip
        r = random.random()
        if r <= .5:
            X[i] = np.flipud(X[i])
            Y[i] = np.flipud(Y[i])
        r = random.random()
        if r <= .5:
            X[i] = np.fliplr(X[i])
            Y[i] = np.fliplr(Y[i])

    # adjust for PyTorch
    X = torch.from_numpy(X)
    X = X.unsqueeze(1)
    Y = torch.from_numpy(Y)
    Y = Y.unsqueeze(1)

    # Batching
    dataset = CustomDataset(X, Y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    return loader, unique, averages, names


def loadVal(noisy_path, pseudo_clean_path):
    Xval = [] # noisy tif stacks
    Yval = [] # pseudoclean targets
    namesVal = [] # names of tifs

    for ind, image_class in enumerate(sorted(os.listdir(noisy_path))): # loop through noisy stacks
        image_path = os.path.join(noisy_path, image_class)
        tif = Image.open(image_path)

        clean_path = os.path.join(pseudo_clean_path, sorted(os.listdir(pseudo_clean_path))[ind]) #get corresponding psuedo-clean
        clean = Image.open(clean_path)
        clean.seek(0)

        for i in range(tif.n_frames):
            tif.seek(i)
            img = np.array(tif)
            Xval.append(np.array(img))
            Yval.append(np.array(clean))
            namesVal.append(image_class)

    Xval = np.array(Xval)
    Yval = np.array(Yval)

    # adjust for PyTorch
    Xval = torch.from_numpy(Xval)
    Xval = Xval.unsqueeze(1)
    Yval = torch.from_numpy(Yval)
    Yval = Yval.unsqueeze(1)
    return Xval, Yval, namesVal


class UNet(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, in_channels=1, out_channels=1):
        """Initializes U-Net."""

        super(UNet, self).__init__()

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv6, upsample5
        self._block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1))
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """

        # Encoder
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block2(pool2)
        pool4 = self._block2(pool3)
        pool5 = self._block2(pool4)

        # Decoder
        upsample5 = self._block3(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block4(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block5(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block5(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block5(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)

        # Final activation
        return self._block6(concat1)


def train(train_path, val_noisy_path, val_pseudo_clean_path, model_save_location, nb_epochs):
    model = UNet()
    optim = Adam(model.parameters())
    criterion = nn.MSELoss()
    bestError = 1000000
    save_epochs = []

    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
        model = model.cuda()
        criterion = criterion.cuda()
    # load validation one time
    Xval, Yval, namesVal = loadVal(val_noisy_path, val_pseudo_clean_path) 
    if use_cuda:
        Xval = Xval.cuda()
        Yval = Yval.cuda()

    for epoch in range(nb_epochs):
        train_loss = 0.0 # Keep track of loss per epoch
        loader, unique, averages, names = loadOne2OneTrain(train_path) # load differently matched images every epoch

        # Train
        for (x, y) in loader:
            model.train()
            if use_cuda:
                x = x.cuda()
                y = y.cuda()
                unique = unique.cuda()

            # Denoise image, track loss, update weights
            denoised = model(x)
            loss = criterion(denoised, y)
            train_loss += loss.item()
            optim.zero_grad()
            loss.backward()
            optim.step()


        # Test Validation
        model.eval()
        y_pred = model(Xval)
        val_MSE = criterion(Yval, y_pred)

        # Adjust for batching
        train_loss = train_loss / len(loader)

        # Visualization with Weights and Biases
        with torch.no_grad():
            # show averages and noisy images
            if epoch == 0:
                for ind, img in enumerate(unique):
                    noisy_display = wandb.Image(img, caption="Noisy Image")
                    wandb.log({os.path.join("N2N Train Images", names[ind]): noisy_display})
                    average_display = wandb.Image(averages[ind], caption="Average")
                    wandb.log({os.path.join("Average Train Image", names[ind]): average_display})

                for ind, img in enumerate(Yval):
                    pseudoclean_display = wandb.Image(Yval.detach().cpu().squeeze(1).numpy(), caption="PseudoClean")
                    wandb.log({os.path.join("PseudoClean (Target)", namesVal[ind]): pseudoclean_display})


            # display model output on train images every __ epoch
            if epoch % (nb_epochs/10) == 0:
                model.eval()
                img = (model(unique).detach().cpu().squeeze(1).numpy())
                for ind, img in enumerate(img):
                    images = wandb.Image(img, caption="Epoch: " + str(epoch))
                    wandb.log({os.path.join("N2N Train Images", names[ind]): images})

                img = y_pred.detach().cpu().squeeze(1).numpy()
                for ind, img in enumerate(img):
                    images = wandb.Image(img, caption="Epoch: " + str(epoch))
                    wandb.log({os.path.join("N2N Val Images", namesVal[ind]): images})

        # Early Stopping Criteria for saving model
        if val_MSE < bestError:
            torch.save(model.state_dict(), model_save_location)
            bestError = val_MSE
            save_epochs.append(epoch)


        # Log loss and criteria in wandb
        wandb.log({
        "Training loss": train_loss,
        "Validation MSE (compared to pseudoclean)": val_MSE ,
        "Epochs Saved" : save_epochs})


if __name__ == "__main__":
    wandb.init(project="Noise2Overfit", config={
            "batch size": "64" ,
            "dataset": '2023-08-30-ReducedTrain',
            "epochs": 1000,
        }
    )

    train('2023-08-30-ReducedTrain', '2023-08-30-noisyTif-Validation', '2023-08-30-pseudoClean-Validation', "2023-08-30-Model-Apr22", 1000)
    wandb.finish()
