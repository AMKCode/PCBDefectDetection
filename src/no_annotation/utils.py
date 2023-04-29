import torch
from torchvision import transforms
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
import cv2
from matplotlib.colors import ListedColormap

NEG_CLASS = 1 # the number that represents a defective PCB
INPUT_IMG_SIZE = (640, 640)

def train(
    dataloader, model, optimizer, criterion, epochs, device, target_accuracy=None
):
    """
    Script to train a model. Returns trained model.
    """
    model.to(device)
    model.train()

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}:", end=" ")
        running_loss = 0
        running_corrects = 0
        n_samples = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            preds_scores = model(inputs)
            preds_class = torch.argmax(preds_scores, dim=-1)
            loss = criterion(preds_scores, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds_class == labels)
            n_samples += inputs.size(0)

        epoch_loss = running_loss / n_samples
        epoch_acc = running_corrects.double() / n_samples
        print("Loss = {:.4f}, Accuracy = {:.4f}".format(epoch_loss, epoch_acc))

        if target_accuracy != None:
            if epoch_acc > target_accuracy:
                print("Early Stopping")
                break

    return model


def evaluate(model, dataloader, device):
    """
    Script to evaluate a model after training.
    Outputs accuracy and balanced accuracy, draws confusion matrix.
    """
    model.to(device)
    model.eval()
    class_names = ['non-defective', 'defective']

    running_corrects = 0
    y_true = np.empty(shape=(0,))
    y_pred = np.empty(shape=(0,))

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        preds_probs = model(inputs)[0]
        preds_class = torch.argmax(preds_probs, dim=-1)

        labels = labels.to("cpu").numpy()
        preds_class = preds_class.detach().to("cpu").numpy()

        y_true = np.concatenate((y_true, labels))
        y_pred = np.concatenate((y_pred, preds_class))

    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    print("Accuracy: {:.4f}".format(accuracy))
    print("Balanced Accuracy: {:.4f}".format(balanced_accuracy))
    print()
    plot_confusion_matrix(y_true, y_pred, class_names=class_names)

def plot_confusion_matrix(y_true, y_pred, class_names="auto"):
    confusion = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=[5, 5])
    
    sns.heatmap(
        confusion,
        annot=True,
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.ylabel("True labels")
    plt.xlabel("Predicted labels")
    plt.title("Confusion Matrix")
    plt.show()

def predict_localize(
    model, dataloader, device, thres=0.8, n_samples=9, show_heatmap=False
):
    """
    Runs predictions for the samples in the dataloader.
    Shows image, its true label, predicted label and probability.
    If an anomaly is predicted, draws bbox around defected region and heatmap.
    """
    model.to(device)
    model.eval()

    class_names = ['non-defective', 'defective']

    n_cols = 3
    n_rows = int(np.ceil(n_samples / n_cols))
    plt.figure(figsize=[n_cols * 5, n_rows * 5])

    counter = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        out = model(inputs)
        probs, class_preds = torch.max(out[0], dim=-1)
        feature_maps = out[1].to("cpu")

        for img_i in range(inputs.size(0)):
            img = inputs[img_i].cpu().detach().numpy()
            img = np.reshape(img, (INPUT_IMG_SIZE[0], INPUT_IMG_SIZE[1])).astype(np.uint8)
            img[np.where(np.logical_and(img>=0, img<=127))] = 0
            img[np.where(np.logical_and(img>=128, img<=255))] = 255

            class_pred = class_preds[img_i]
            prob = probs[img_i]
            label = labels[img_i]
            heatmap = feature_maps[img_i][NEG_CLASS].detach().numpy()

            counter += 1
            plt.subplot(n_rows, n_cols, counter)
            clrs = ['black', 'red', 'white']
            if (class_pred != NEG_CLASS):
                plt.imshow(img, cmap=ListedColormap(clrs))
            plt.axis("off")
            plt.title(
                "Predicted: {}, Prob: {:.3f}, True Label: {}".format(
                    class_names[class_pred], prob, class_names[label]
                )
            )

            if class_pred == NEG_CLASS:
                heat_img = ((heatmap > thres) * 255).astype(np.uint8)
                thresh = cv2.threshold(heat_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                for c in cnts:
                    x,y,w,h = cv2.boundingRect(c)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (120, 120,12), 2)
                plt.imshow(img, cmap=ListedColormap(clrs))
                if show_heatmap:
                    plt.imshow(heatmap, cmap="Reds", alpha=0.3)
            if counter == n_samples:
                plt.tight_layout()
                plt.show()
                return
