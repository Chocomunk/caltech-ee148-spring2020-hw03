import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from models import Net

if __name__=="__main__":
    # "input arguments"
    data_dir = 'data/'
    model_dir = 'out/best_full_train/mnist_model.pt'
    test_batch_size = 1000

    # Config variables
    assert os.path.exists(model_dir)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Set the test model
    print("Loading model...")
    model = Net().to(device)
    model.load_state_dict(torch.load(model_dir))

    # Register hook to extract features
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    model.fc3.register_forward_hook(get_activation('features'))

    # Load test data
    test_dataset = datasets.MNIST(data_dir, train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=True, **kwargs)

    # Process test set
    print("Processing test set...")
    model.eval()    # Set the model to inference mode
    num_incorrect = 0
    incorrect = []
    conf_mat = np.zeros((10,10))    # (predicted, true)
    features = [[] for _ in range(10)]  # (class, sample, *features)
    feat_imgs = [[] for _ in range(10)]
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            # Move to numpy before-hand for faster access
            p = preds.cpu().numpy()
            t = target.view_as(preds).cpu().numpy()
            d = data.cpu().numpy()
            feats = activation['features'].cpu().numpy()

            # Process model results
            for i in range(len(p)):
                # Find and store incorrect predictions
                if num_incorrect < 9:
                    if p[i][0] != t[i][0]:
                        incorrect.append((len(feat_imgs[t[i][0]]), p[i][0], t[i][0]))
                        num_incorrect += 1
                
                # Populate confusion matrix
                conf_mat[p[i][0], t[i][0]] += 1

                # Store feature vector
                features[t[i][0]].append(feats[i])
                feat_imgs[t[i][0]].append(d[i][0,:,:])

    # Transform feature vectors (ordered by class)
    class_sizes = [len(e) for e in features]
    features_tf = np.zeros((10000, 64))
    i = 0
    for class_feats in features:
        for feat in class_feats:
            features_tf[i] = feat
            i += 1
    features_tf = TSNE(n_components=2).fit_transform(features_tf)

    # Finding closest examples
    def img_indices(idx):
        """ Find 2d index of image """
        img_ind = idx
        class_ind = 0
        while img_ind > class_sizes[class_ind]:
            img_ind -= class_sizes[class_ind]
            class_ind += 1
        return class_ind, img_ind

    similar_imgs = np.zeros((4,9,2), dtype=int)     # Stores (class, img)
    for i in range(4):
        center = np.random.randint(10000)
        similar_imgs[i,0] = img_indices(center)

        # Sort by distance to center
        dists = np.hypot(*(features_tf - features_tf[center]).T)
        ind_sort = np.argsort(dists)

        # Store closest neighbors
        j = 0
        k = 1
        while k < 9:
            if ind_sort[j] != center:
                similar_imgs[i,k] = img_indices(ind_sort[j])
                k += 1
            j += 1


    print("Generating visualizations...")

    # ----------------- Visualize first 9 incorrect examples ----------------- 
    fig, axs = plt.subplots(3,3)
    fig.suptitle("Incorrect Predictions in Test Set")
    for i, (img_idx, pred, targ) in enumerate(incorrect[:9]):
        a, b = i//3, i%3
        axs[a,b].imshow(feat_imgs[targ][img_idx])
        axs[a,b].axis('off')
        axs[a,b].title.set_text("Pred: {}, True: {}".format(pred, targ))
    plt.savefig("incorrect_preds.png")
    plt.show()
    

    # ----------------- Visualize first convolution kernels ----------------- 
    conv1 = model.conv1.weight.data.cpu().numpy()
    fig, axs = plt.subplots(4,4)
    fig.suptitle("First Conv Layer Kernels")
    for i, conv1 in enumerate(conv1):
        a, b = i//4, i%4
        axs[a,b].imshow(conv1[0,:,:])
        axs[a,b].axis('off')
        axs[a,b].title.set_text("Kernel: {}".format(i))
    plt.savefig("conv1_kernels.png")
    plt.show()

    # -------------------- Visualize confusion matrix -------------------- 
    fig, ax = plt.subplots()
    im = ax.imshow(conf_mat)
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xticklabels(np.arange(10))
    ax.set_yticklabels(np.arange(10))
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right")

    # Loop over data dimensions and create text annotations.
    for i in range(10):
        for j in range(10):
            if conf_mat[i,j] != 0:
                text = ax.text(j, i, int(conf_mat[i, j]),
                            ha="center", va="center", color="w")

    ax.set_title("Test Set Confusion Matrix (10000 examples)")
    fig.tight_layout()
    plt.savefig("confusion_mat.png")
    plt.show()

    # --------------- Visualize TSNE-tranformed feature vectors --------------- 
    last_ind = 0
    for i, size in enumerate(class_sizes):
        plt.scatter(features_tf[last_ind:last_ind+size, 0], 
                    features_tf[last_ind:last_ind+size, 1], 
                    label="Digit: {}".format(i))
        last_ind += size
    plt.legend()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("TSNE-transformed Test Set Feature Vectors")
    plt.savefig("tsne_feats.png")
    plt.show()

    # --------------- Visualize Similar Images --------------- 
    fig, axs = plt.subplots(4,9)
    fig.suptitle("Similar-Feature Images")
    for i in range(4):
        for j in range(9):
            class_ind, img_ind = similar_imgs[i,j]
            axs[i,j].imshow(feat_imgs[class_ind][img_ind])
            axs[i,j].axis('off')
            axs[i,j].title.set_text("{}".format(class_ind))
    plt.savefig("similar_imgs.png")
    plt.show()
