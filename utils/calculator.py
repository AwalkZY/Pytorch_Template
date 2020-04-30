import torch
import numpy as np


def calculate_dist(x, y, dist):
    assert x.size(-1) == y.size(-1), "Incompatible dimension of two matrix! {} and {} are given. ".method(x.size(),
                                                                                                          y.size())
    assert dist in ['Euclid', 'Cosine'], "Invalid distance criterion!"
    d = x.size(-1)
    x = x.contiguous().view(-1, d)
    y = y.contiguous().view(-1, d)
    m, n = x.size(0), y.size(0)
    # x : in shape (m,d), y : (n,d)
    x = torch.unsqueeze(x, dim=1).expand(m, n, d)
    y = torch.unsqueeze(y, dim=0).expand(m, n, d)
    if dist == 'Euclid':
        return torch.dist(x, y, p=2)
    elif dist == 'Cosine':
        return 1 - torch.cosine_similarity(x, y, dim=2)  # shape: (m, n)


def get_mIoU(iou):
    data = dict([('IoU@0.%d' % i, 0.0) for i in range(1, 10)])
    data['mIoU'] = np.mean(iou)
    for i in range(1, 10):
        data['IoU@0.%d' % i] = np.mean(iou >= 0.1 * i)
    return data


def calculate_iou3d(box0, box1):
    """
    :param box0: in shape (box_numbers, height_step, 4)
    :param box1: in shape (box_numbers, height_step, 4)
    :return: vIoU of two list of boxes
    """
    intersections = calculate_intersection2d(box0, box1).view(box0.size()[:-1]).sum(dim=1)
    unions = calculate_union2d(box0, box1).view(box0.size()[:-1]).sum(dim=1)
    return intersections / unions


def calculate_intersection2d(box0, box1, method="Corner_Corner"):
    assert method in ["Corner_Corner", "Corner_Length"], "Invalid Format of Measurement!"
    box0 = box0.contiguous().view(-1, 4)
    box1 = box1.contiguous().view(-1, 4)
    if method == "Corner_Length":
        l0, t0, r0, b0 = box0[:, 0], box0[:, 1], box0[:, 0] + box0[:, 2], box0[:, 1] + box0[:, 3]
        l1, t1, r1, b1 = box1[:, 0], box1[:, 1], box1[:, 0] + box1[:, 2], box1[:, 1] + box1[:, 3]
    elif method == "Corner_Corner":
        l0, t0, r0, b0 = box0[:, 0], box0[:, 1], box0[:, 2], box0[:, 3]
        l1, t1, r1, b1 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    width = torch.min(r0, r1) - torch.max(l0, l1)
    width[width < 0] = 0
    height = torch.min(b0, b1) - torch.max(t0, t1)
    height[height < 0] = 0
    intersection = width * height
    return intersection


def calculate_union2d(box0, box1, method="Corner_Corner"):
    box0 = box0.contiguous().view(-1, 4)
    box1 = box1.contiguous().view(-1, 4)
    if method == "Corner_Length":
        l0, t0, r0, b0 = box0[:, 0], box0[:, 1], box0[:, 0] + box0[:, 2], box0[:, 1] + box0[:, 3]
        l1, t1, r1, b1 = box1[:, 0], box1[:, 1], box1[:, 0] + box1[:, 2], box1[:, 1] + box1[:, 3]
    elif method == "Corner_Corner":
        l0, t0, r0, b0 = box0[:, 0], box0[:, 1], box0[:, 2], box0[:, 3]
        l1, t1, r1, b1 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    width = torch.min(r0, r1) - torch.max(l0, l1)
    width[width < 0] = 0
    height = torch.min(b0, b1) - torch.max(t0, t1)
    height[height < 0] = 0
    union = (b0 - t0) * (r0 - l0) + (b1 - t1) * (r1 - l1) - width * height + 1e-15
    return union


def calculate_iou2d(box0, box1):
    return calculate_intersection2d(box0, box1) / calculate_union2d(box0, box1)


def calculate_iou1d(pred_first, pred_last, true_first, true_last):
    """
        calculate temporal intersection over union
    """
    if type(pred_first) is torch.Tensor:
        pred_first = pred_first.cpu().numpy()
        pred_last = pred_last.cpu().numpy()
        true_first = true_first.cpu().numpy()
        true_last = true_last.cpu().numpy()
    elif type(pred_first) is list:
        pred_first = np.array(pred_first)
        pred_last = np.array(pred_last)
        true_first = np.array(true_first)
        true_last = np.array(true_last)
    pred_first, pred_last = pred_first.astype(float), pred_last.astype(float)
    true_first, true_last = true_first.astype(float), true_last.astype(float)
    union = (np.min(np.stack([pred_first, true_first], 0), 0), np.max(np.stack([pred_last, true_last], 0), 0))
    inter = (np.max(np.stack([pred_first, true_first], 0), 0), np.min(np.stack([pred_last, true_last], 0), 0))
    iou = 1.0 * (inter[1] - inter[0] + 1) / (union[1] - union[0] + 1)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return torch.tensor(iou)


def find_intersection(set_1, set_2):
    """
    Corner-Corner
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """
    Corner-Corner
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    set_1 = set_1.float()
    set_2 = set_2.float()
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)
    return intersection / union  # (n1, n2)


def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, n_classes):
    """
    Calculate the Mean Average Precision (mAP) of detected objects.
    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation
    :param n_classes: number of classes
    :param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
    :param det_labels: list of tensors, one tensor for each image containing detected objects' labels
    :param det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
    :param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
    :param true_labels: list of tensors, one tensor for each image containing actual objects' labels
    :param true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
    :return: list of average precisions for all classes, mean average precision (mAP)
    """
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(
        true_boxes) == len(true_labels) == len(true_difficulties)
    # these are all lists of tensors of the same length, i.e. number of images

    device = det_boxes[0].device
    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.tensor(true_images).int().to(device)
    # (n_objects), n_objects is the total no. of objects across all images
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.tensor(det_images).int().to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float).to(device)  # (n_classes - 1)
    # here we suppose c=0 corresponds background.
    for c in range(1, n_classes):
        # Extract only objects with this class
        # (n_class_objects) means the number of objects in this class
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
        n_easy_class_objects = (1 - true_class_difficulties).sum().item()  # ignore difficult objects

        # Keep track of which true objects with this class have already been 'detected'
        # So far, none
        true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(device)
        # (n_class_objects)

        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue

        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        true_positives = torch.zeros(n_class_detections, dtype=torch.float).to(device)  # (n_class_detections)
        false_positives = torch.zeros(n_class_detections, dtype=torch.float).to(device)  # (n_class_detections)
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar
            # Find objects in the same image with this class, their difficulties,
            # and whether they have been detected before
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image of this class
            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            original_ind = torch.tensor(range(true_class_boxes.size(0))).int()[true_class_images == this_image][ind]
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if max_overlap.item() > 0.5:
                # If the object it matched with is 'difficult', ignore it
                if object_difficulties[ind] == 0:
                    # If this object has already not been detected, it's a true positive
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                    # Otherwise, it's a false positive (since this object is already accounted for)
                    else:
                        false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1

        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumulative_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumulative_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumulative_precision = cumulative_true_positives / (
                cumulative_true_positives + cumulative_false_positives + 1e-10)  # (n_class_detections)
        cumulative_recall = cumulative_true_positives / n_easy_class_objects  # (n_class_detections)

        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumulative_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumulative_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean().item()

    # Keep class-wise average precisions in a dictionary
    # average_precisions = {rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}
    return mean_average_precision
    # return average_precisions, mean_average_precision


if __name__ == "__main__":
    # a1 = torch.tensor([[0, 0, 1, 2], [2, 3, 4, 5]])
    # a2 = torch.tensor([[0, 0, 3, 6]])
    # print(find_jaccard_overlap(a1, a2))
    pred_head = [0, 2]
    pred_tail = [1, 3]
    true_head = [0, 3]
    true_tail = [1, 4]
    print(calculate_iou1d(pred_head, pred_tail,
                          true_head, true_tail))