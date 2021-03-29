import json
import os
from bz2 import BZ2File

def check_dataset():
    with open("/media/D/image_chat/data/train.json", "r", encoding="utf-8") as f:
        # dict_keys(['dialog', 'image_hash'])
        print("======================================")
        print("train:")
        data = json.load(f)
        print(data[0].keys())

    with open("/media/D/image_chat/data/valid.json", "r", encoding="utf-8") as f:
        # dict_keys(['dialog', 'candidates', 'image_hash'])
        print("======================================")
        print("valid:")
        data = json.load(f)
        print(data[0].keys())
        print("dialog")
        print(data[0]["dialog"])
        print("candidates:")
        print(len(data[0]["candidates"]))
        for i in range(len(data[0]["candidates"])):
            print(i)
            for key in data[0]["candidates"][i]:
                print(key, len(data[0]["candidates"][i][key]))


    with open("/media/D/image_chat/data/test.json", "r", encoding="utf-8") as f:
        # dict_keys(['dialog', 'candidates', 'image_hash'])
        print("======================================")
        print("test:")
        data = json.load(f)
        print(data[0].keys())
        print("dialog")
        print(data[0]["dialog"])
        print("candidates:")
        print(len(data[0]["candidates"]))
        for i in range(len(data[0]["candidates"])):
            print(i)
            for key in data[0]["candidates"][i]:
                print(key, len(data[0]["candidates"][i][key]))



def check_bz2():
    image_root = "/media/D/image_chat/data/images"
    images = os.listdir(image_root)
    hashes = []
    for dt in ["train", "val", "test"]:
        with open(os.path.join("/media/D/image_chat/data", "%s.json" % dt)) as f:
            data = json.load(f)
            hashes += [d["image_hash"] for d in data]

    original_images = ["{}.jpg".format(h) for h in hashes]
    downloaded = set(images)
    missed_files = []
    print("{} / {}".format(len(images), len(original_images)))
    for image in original_images:
        if image not in downloaded:
            # print(image)
            missed_files.append(image.split(".")[0])
    print("missed files:", len(missed_files), missed_files)

    to_download = []
    lines = []
    file = "/media/D/ParlAI/yfcc100m_dataset.bz2"
    with BZ2File(file, "r") as bzfin:
        for i, line in enumerate(bzfin):
            tmp_hash = line.split()[2].decode("utf-8")
            tmp_url = line.split()[15].decode("utf-8")
            if tmp_hash in missed_files:
                lines.append(line.decode("utf-8"))
                to_download.append(tmp_url)
            if i % 1000000 == 0:
                print("{} %".format(i / 1000000), end="\t")
                print("detect {} files".format(len(to_download)))
                if len(lines) > 0:
                    print(lines[-1])
    print("to download: ", to_download)
    with open("tmp_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def check_downloaded():
    image_root = "/home1/zhaoxl/image_chat/data/images"
    images = os.listdir(image_root)
    hashes = []
    for dt in ["train", "val", "test"]:
        with open(os.path.join("/home1/zhaoxl/image_chat/data", "%s.json" % dt)) as f:
            data = json.load(f)
            hashes += [d["image_hash"] for d in data]

    original_images = ["{}.jpg".format(h) for h in hashes]
    urls = ["{}/{}/{}/{}.jpg".format("https://multimedia-commons.s3-us-west-2.amazonaws.com/data/images", h[:3], h[3:6], h) for h in hashes]
    downloaded = set(images)
    todo_url = []
    print("{} / {}".format(len(images), len(original_images)))
    for image, url in zip(original_images, urls):
        if image not in downloaded:
            print(image)
            # if not image.startswith("ac8"):
            todo_url.append(url)
            # todo_url.append("{}/{}/{}.jpg".format(image.split(".")[0][:3],
            #                                          image.split(".")[0][3:6],
            #                                          image.split(".")[0][:],))
    print("todo url: ", todo_url)


def check_url():
    tgt_dir = "/home1/zhaoxl/image_chat/data/images"
    with open("tmp_log.txt", "r", encoding="utf-8") as f:
        for line in f.readlines():
            if len(line.strip()) == 0:
                continue
            parts = line.strip().split()
            for info in parts[::-1]:
                if info.endswith(".jpg") and info.startswith("http:"):
                    url = info
                    # print(url)
                    # image_name = url.split("/")[-1]
                    break
            hash = parts[2]
            print(hash, url)
            os.system("wget {}".format(url))
            os.system("mv {} {}".format(url.split("/")[-1], "{}/{}.jpg".format(tgt_dir, hash)))
            # os.system("rm {}".format(os.path.join(tgt_dir, image_name)))

def get_image_list():
    image_list = os.listdir("/home1/zhaoxl/image_chat/data/images")
    with open("/home1/zhaoxl/image_chat/data/image_list.json", "w", encoding="utf-8") as f:
        json.dump(image_list, f)

if __name__ == "__main__":
    # check_bz2()
    # check_dataset()
    # check_url()
    check_downloaded()
    # get_image_list()