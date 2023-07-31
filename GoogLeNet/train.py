import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

from Model import GoogleNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device for training.".format(device))

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    img_root = "D:/BaiduNetdiskDownload/DataSets/flower_data"
    train_dataset = datasets.ImageFolder(root=os.path.join(img_root,"train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    val_dataset = datasets.ImageFolder(root=os.path.join(img_root,"val"),
                                       transform=data_transform["val"])
    val_num = len(val_dataset)

    flower_list = train_dataset.class_to_idx
    # 倒置val、key方便预测结果的显示
    cla_dict = dict((val, key) for key, val in flower_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    # 使用 nw 变量可以在 DataLoader 中设置 num_workers 参数，从而控制数据加载过程的并行度，以提高数据加载的效率。
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print("Using {} dataloader workers every process".format(nw))

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=nw)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=nw)
    print("Using {} images for training, using {} images for validation.".format(train_num, val_num))

    net = GoogleNet(num_classes=5, aux_logits=True, init_weights=True)
    # 如果要使用官方的预训练权重，注意是将权重载入官方的模型，不是我们自己实现的模型
    # 官方的模型中使用了bn层以及改了一些参数，不能混用
    # import torchvision
    # net = torchvision.models.googlenet(num_classes=5)
    # model_dict = net.state_dict()
    # # 预训练权重下载地址: https://download.pytorch.org/models/googlenet-1378be20.pth
    # pretrain_model = torch.load("googlenet.pth")
    # del_list = ["aux1.fc2.weight", "aux1.fc2.bias",
    #             "aux2.fc2.weight", "aux2.fc2.bias",
    #             "fc.weight", "fc.bias"]
    # pretrain_dict = {k: v for k, v in pretrain_model.items() if k not in del_list}
    # model_dict.update(pretrain_dict)
    # net.load_state_dict(model_dict)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0003)

    epochs = 20
    best_acc = 0.0
    save_path = './GoogleNet.pth'
    train_steps = len(train_dataloader)  # train_dataloader = 数据总数  / batch_size
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0  # 总损失
        train_bar = tqdm(train_dataloader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits, aux2_loss, aux1_loss = net(images.to(device))
            loss_1 = loss_function(logits, labels.to(device))
            loss_2 = loss_function(aux1_loss, labels.to(device))
            loss_3 = loss_function(aux2_loss, labels.to(device))
            loss = loss_1 + loss_2 * 0.3 + loss_3 * 0.3  # 原论文中辅助分类器的权重为0.3
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch+1,
                                                                     epochs,
                                                                     loss)
        # validate
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_dataloader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]  # 返回每个batch中最大值的索引，也就是预测的label
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print("[epoch %d] train_mean_loss: %.3f val_acc: %.3f" %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
    print("Finished Training")


if __name__ == "__main__":
    main()
