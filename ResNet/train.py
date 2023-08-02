import os
import torch
import json
import torch.nn as nn

from torchvision import datasets, transforms
from model import resnet34


def main(batch_size=32, epochs=2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} for training.".format(device))

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    data_root = "D:/BaiduNetdiskDownload/DataSets/flower_data"
    train_dataset = datasets.ImageFolder(os.path.join(data_root, "train"),
                                         transform=data_transform["train"])
    train_data_num = len(train_dataset)
    val_dataset = datasets.ImageFolder(os.path.join(data_root, "val"),
                                       transform=data_transform["val"])
    val_data_num = len(val_dataset)
    print("using {} images fot training, {} images for validation.".format(
        train_data_num, val_data_num
    ))

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_dict = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_dict.items())
    # write dict to json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print("using {} dataloader workers every process.\n".format(num_workers))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)

    # 此时由于迁移训练使用官方的预训练参数，因此暂时不指定num_classes
    net = resnet34()
    resnet34_weight_path = './model_hub/resnet34.pth'
    assert os.path.exists(resnet34_weight_path), "file {} does not exist.".format(resnet34_weight_path)
    # net.load_state_dict(torch.load(resnet34_weight_path, map_location='cpu'))
    missing_keys, unexpected_keys = net.load_state_dict(torch.load(resnet34_weight_path, map_location='cpu'), strict=False)
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 5)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0001)

    best_acc = 0.0
    save_path = './checkpoints/Trained_ResNet34.pth'
    train_steps = len(train_loader)

    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            rate = (step+1)/len(train_loader)
            a = "*" * int(rate*50)
            b = "." * int((1-rate) * 50)
            print("\rtrain_loss: {:^3.0f}%[{}->{}]{:.3f}".format(
                int(rate*100), a, b, loss
            ), end="")
        print()
        print("training processes finished.")

        # val
        net.eval()
        acc = 0.0
        with torch.no_grad():
            for step, val_data in enumerate(val_loader, start=0):
                val_imaged, val_labels = val_data
                outputs = net(val_imaged.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                rate = (step + 1) / len(val_loader)
                a = "*" * int(rate * 50)
                b = "." * int((1 - rate) * 50)
                print("\rAcc: {:^3.0f}%[{}->{}]{:.3f}".format(
                    int(rate * 100), a, b, acc
                ), end="")
            print()
            print("validation processes finished.")

        val_accurate = acc / val_data_num
        print("[epoch {}]\ttrain_loss: {:.3f}\tval_accuracy: {:.3f}\n".format(
            epoch+1, running_loss / train_steps, val_accurate
        ))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print("Finished Training.")


if __name__ == "__main__":
    batch_size = 32
    epochs = 2
    main(batch_size, epochs)
