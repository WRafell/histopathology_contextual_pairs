import torch


def two_distortion_positive_pairs(batch: list):
    views = batch[0]
    return views[0], views[1]


def shuffle_positive_pairs(batch: list):
    views_1, views_2 = batch[0]
    if torch.rand(1).item() > 0.5:
        shuffle_indeces = torch.randperm(views_2.size(0))
        views_2 = views_2[shuffle_indeces]
    return views_1, views_2


def closest_patch_positive_pairs(batch: list):
    # batch: ['views', 'img_path', 'slide_name', 'coord']
    views_1, views_2 = batch[0]
    slide_names = batch[2]
    rows, cols = batch[3]
    coords = torch.stack([rows, cols], dim=1).float()
    batch_size = len(slide_names)

    same_slide_matrix = torch.tensor([i==j for i in slide_names for j in slide_names]).reshape((batch_size,batch_size))
    euclidian_distance_matrix = torch.cdist(coords, coords)

    neighbor_coords = []
    for i in range(batch_size):
        same_slides = same_slide_matrix[i]
        if same_slides.sum() == 1:
            # If there is no other patch from the same WSI, skip
            neighbor_coords.append(i)
            continue
        different_slides = ~ same_slides
        euclidian_distances = euclidian_distance_matrix[i]
        euclidian_distances[different_slides] = float('inf')
        _, indices = torch.topk(euclidian_distances, k=2, largest=False)
        neighbor_coords.append(indices[1].item()) # ignore same patch
    views_2 = views_2[neighbor_coords]
    return views_1, views_2
    

        

