import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import json
from prettytable import PrettyTable
from torchvision import transforms, datasets
from GoogLeNet.Model import GoogleNet


class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for preds, tru in zip(preds, labels):
            self.matrix[preds, tru] +=1

    def summary(self):
        # calculate Acc
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        ACC = sum_TP / np.sum(self.matrix)
        print("the model acc is : ", ACC)

        # calculate precision, recall, specificity
        tabel = PrettyTable()
        tabel.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            tabel.add_row([self.labels[i], Precision, Recall, Specificity])
        print(tabel)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        plt.yticks(range(self.num_classes), self.labels)
        plt.colorbar
        plt.xlabel("True Labels")
        plt.ylabel("Predicted Labels")
        plt.title("Confusion Matrix")

        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        # 调整图像为紧凑布局
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    '''
        此处使用GoogLeNet作为示例：保存验证集前处理阶段相同
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} for training.".format(device))

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_root = "D:/BaiduNetdiskDownload/DataSets/flower_data"
    val_dataset = datasets.ImageFolder(os.path.join(data_root, "val"),
                                       transform=data_transform)
    batch_size = 32
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)

    # net = ResNet34(num_classes=5)
    net = GoogleNet(num_classes=5, aux_logits=False).to(device)
    weight_pth = "../GoogLeNet/GoogleNet.pth"
    missing_keys, unexpected_keys = net.load_state_dict(torch.load(weight_pth, map_location=device), strict=False)

    json_label_pth = "./class_indices.json"
    json_file = open(json_label_pth, 'r')
    class_indict = json.load(json_file)
    labels = [label for _, label in class_indict.items()]

    confusion = ConfusionMatrix(num_classes=5, labels=labels)
    net.eval()
    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to('cpu').numpy(), val_labels.to('cpu').numpy())
    confusion.plot()
    confusion.summary()
