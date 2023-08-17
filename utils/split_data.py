import os
import random


def main():
    """
        对于传统机器学习阶段（数据集在万这个数量级），一般分配比例为训练集和测试集的比例为7:3或是8:2。
        为了进一步降低信息泄露同时更准确的反应模型的效能，更为常见的划分比例是训练集、验证集、测试的比例为6：2：2。

        而大数据时代，这个比例就不太适用了。因为百万级的数据集，即使拿1%的数据做test也有一万之多，已经足够了。
        可以拿更多的数据做训练。因此常见的比例可以达到98：1：1，甚至可以达到99.5：0.3：0.2等。
    """
    # random.seed(0)

    file_path = "D:/BaiduNetdiskDownload/DataSets/seaships/Annotations"
    assert os.path.exists(file_path), f"path: {file_path} is not exists."

    val_rate = 0.2
    test_rate = 0.2

    file_name = sorted([file.split(".")[0] for file in os.listdir(file_path)])
    file_num = len(file_name)
    file_mid = file_num // 2  # 在前半段选取val，后半段选取test
    val_index = random.sample(range(0, file_mid), k=int(val_rate * file_num))
    test_index = random.sample(range(file_mid, file_num), k=int(test_rate * file_num))

    train_files = []
    val_files = []
    test_files = []

    for index, file_name in enumerate(file_name):
        if index in val_index:
            val_files.append(file_name)
        elif index in test_index:
            test_files.append(file_name)
        else:
            train_files.append(file_name)

    try:
        train_file_save = open("train.txt", "w")
        val_file_save = open("val.txt", "w")
        test_file_save = open("test.txt", "w")
        train_file_save.write("\n".join(train_files))
        val_file_save.write("\n".join(val_files))
        test_file_save.write("\n".join(test_files))

    except FileExistsError as e:
        print(e)
        exit(1)


if __name__ == "__main__":
    main()
