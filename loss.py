import paddle


def split(x, num_or_sections, axis=0):
    if isinstance(num_or_sections, int):
        return paddle.split(x, x.shape[axis] // num_or_sections, axis)
    else:
        return paddle.split(x, num_or_sections, axis)


def min(*args, **kwargs):
    if "input" in kwargs:
        kwargs["x"] = kwargs.pop("input")

    out_v = None
    if "out" in kwargs:
        out_v = kwargs.pop("out")

    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.minimum(*args, **kwargs)
    elif len(args) == 2 and isinstance(args[1], paddle.Tensor):
        ret = paddle.minimum(*args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 2:
            if out_v:
                ret = paddle.min(*args, **kwargs), paddle.argmin(*args, **kwargs)
                paddle.assign(ret[0], out_v[0])
                paddle.assign(ret[1], out_v[1])
                return out_v
            else:
                ret = paddle.min(*args, **kwargs), paddle.argmin(*args, **kwargs)
                return ret
        else:
            ret = paddle.min(*args, **kwargs)
            return ret

    if out_v:
        paddle.assign(ret, out_v)
        return out_v
    else:
        return ret


def max(*args, **kwargs):
    if "input" in kwargs:
        kwargs["x"] = kwargs.pop("input")

    out_v = None
    if "out" in kwargs:
        out_v = kwargs.pop("out")

    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.maximum(*args, **kwargs)
    elif len(args) == 2 and isinstance(args[1], paddle.Tensor):
        ret = paddle.maximum(*args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 2:
            if out_v:
                ret = paddle.max(*args, **kwargs), paddle.argmax(*args, **kwargs)
                paddle.assign(ret[0], out_v[0])
                paddle.assign(ret[1], out_v[1])
                return out_v
            else:
                ret = paddle.max(*args, **kwargs), paddle.argmax(*args, **kwargs)
                return ret
            return out_v
        else:
            ret = paddle.max(*args, **kwargs)
            return ret

    if out_v:
        paddle.assign(ret, out_v)
        return out_v
    else:
        return ret


def get_dice_loss(gt_score, pred_score):
    inter = paddle.sum(x=gt_score * pred_score)
    union = paddle.sum(x=gt_score) + paddle.sum(x=pred_score) + 1e-05
    return 1.0 - 2 * inter / union


def get_geo_loss(gt_geo, pred_geo):
    d1_gt, d2_gt, d3_gt, d4_gt, angle_gt = split(x=gt_geo, num_or_sections=1, axis=1)
    d1_pred, d2_pred, d3_pred, d4_pred, angle_pred = split(
        x=pred_geo, num_or_sections=1, axis=1
    )
    area_gt = (d1_gt + d2_gt) * (d3_gt + d4_gt)
    area_pred = (d1_pred + d2_pred) * (d3_pred + d4_pred)
    w_union = min(d3_gt, d3_pred) + min(d4_gt, d4_pred)
    h_union = min(d1_gt, d1_pred) + min(d2_gt, d2_pred)
    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect
    iou_loss_map = -paddle.log(x=(area_intersect + 1.0) / (area_union + 1.0))
    angle_loss_map = 1 - paddle.cos(x=angle_pred - angle_gt)
    return iou_loss_map, angle_loss_map


class Loss(paddle.nn.Layer):
    def __init__(self, weight_angle=10):
        super(Loss, self).__init__()
        self.weight_angle = weight_angle

    def forward(self, gt_score, pred_score, gt_geo, pred_geo, ignored_map):
        if paddle.sum(x=gt_score) < 1:
            return paddle.sum(x=pred_score + pred_geo) * 0
        classify_loss = get_dice_loss(gt_score, pred_score * (1 - ignored_map))
        iou_loss_map, angle_loss_map = get_geo_loss(gt_geo, pred_geo)
        angle_loss = paddle.sum(x=angle_loss_map * gt_score) / paddle.sum(x=gt_score)
        iou_loss = paddle.sum(x=iou_loss_map * gt_score) / paddle.sum(x=gt_score)
        geo_loss = self.weight_angle * angle_loss + iou_loss
        print(
            "classify loss is {:.8f}, angle loss is {:.8f}, iou loss is {:.8f}".format(
                classify_loss, angle_loss, iou_loss
            )
        )
        return geo_loss + classify_loss
