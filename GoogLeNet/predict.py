import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from Model import GoogleNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using: {} for predict.".format(device))

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # load image
    img_path = "./R.jpg"
    assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    img = data_transform(img)
    # img size : c h w ---> n c h w
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = GoogleNet(num_classes=5, aux_logits=False).to(device)

    # load model weight
    weight_path = './GoogleNet.pth'
    assert os.path.exists(weight_path), "file: '{}' does not exist.".format(weight_path)
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)

    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        # print(output)
        predict = torch.softmax(output, dim=0)
        # print(predict)
        predict_cla = torch.argmax(predict).numpy()
        # print(predict_cla)

    print_res = "class: {}    prob:{:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)

    for i in range(len(predict)):
        print("class: {:10}    prob: {:.3}".format(class_indict[str(i)],
                                                   predict[i].numpy()))
    plt.show()


if __name__ == "__main__":
    main()
