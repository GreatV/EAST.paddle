import paddle
from dataset import custom_dataset
from model import EAST
from loss import Loss
import os
import time


def train(
    train_img_path,
    train_gt_path,
    pths_path,
    batch_size,
    lr,
    num_workers,
    epoch_iter,
    interval,
):
    file_num = len(os.listdir(train_img_path))
    trainset = custom_dataset(train_img_path, train_gt_path)
    train_loader = paddle.io.DataLoader(
        dataset=trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    criterion = Loss()
    device = str("cuda:0" if paddle.device.cuda.device_count() >= 1 else "cpu").replace(
        "cuda", "gpu"
    )
    model = EAST()
    data_parallel = False
    if paddle.device.cuda.device_count() > 1:
        model = paddle.DataParallel(layers=model)
        data_parallel = True
    model.to(device)
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(), learning_rate=lr, weight_decay=0.0
    )
    tmp_lr = paddle.optimizer.lr.MultiStepDecay(
        milestones=[epoch_iter // 2], gamma=0.1, learning_rate=optimizer.get_lr()
    )
    optimizer.set_lr_scheduler(tmp_lr)
    scheduler = tmp_lr
    for epoch in range(epoch_iter):
        model.train()
        scheduler.step()
        epoch_loss = 0
        epoch_time = time.time()
        for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
            start_time = time.time()
            img, gt_score, gt_geo, ignored_map = (
                img.to(device),
                gt_score.to(device),
                gt_geo.to(device),
                ignored_map.to(device),
            )
            pred_score, pred_geo = model(img)
            loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
            epoch_loss += loss.item()
            optimizer.clear_grad()
            loss.backward()
            optimizer.step()
            print(
                "Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}".format(
                    epoch + 1,
                    epoch_iter,
                    i + 1,
                    int(file_num / batch_size),
                    time.time() - start_time,
                    loss.item(),
                )
            )
        print(
            "epoch_loss is {:.8f}, epoch_time is {:.8f}".format(
                epoch_loss / int(file_num / batch_size), time.time() - epoch_time
            )
        )
        print(time.asctime(time.localtime(time.time())))
        print("=" * 50)
        if (epoch + 1) % interval == 0:
            state_dict = (
                model.module.state_dict() if data_parallel else model.state_dict()
            )
            paddle.save(
                obj=state_dict,
                path=os.path.join(pths_path, "model_epoch_{}.pth".format(epoch + 1)),
            )


if __name__ == "__main__":
    train_img_path = os.path.abspath("../ICDAR_2015/train_img")
    train_gt_path = os.path.abspath("../ICDAR_2015/train_gt")
    pths_path = "./pths"
    batch_size = 24
    lr = 0.001
    num_workers = 4
    epoch_iter = 600
    save_interval = 5
    train(
        train_img_path,
        train_gt_path,
        pths_path,
        batch_size,
        lr,
        num_workers,
        epoch_iter,
        save_interval,
    )
