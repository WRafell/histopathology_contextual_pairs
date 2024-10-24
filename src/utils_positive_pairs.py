import torch


def two_distortion_positive_pairs(batch: list) -> tuple:
    views_1, views_2 = batch[0]
    return views_1, views_2


def shuffle_positive_pairs(batch: list) -> tuple:
    views_1, views_2 = batch[0]
    if torch.rand(1).item() > 0.5:
        shuffle_indeces = torch.randperm(views_2.size(0))
        views_2 = views_2[shuffle_indeces]
    return views_1, views_2


def closest_patch_positive_pairs(batch: list) -> tuple:
    """
    Finds the closest positive patch pairs within the same slide for a batch of image patches.

    This function takes a batch of image patches and their associated metadata, and for each patch,
    it finds the closest neighboring patch from the same slide. If a patch has no neighbors from
    the same slide, it is paired with itself.

    Parameters:
    -----------
    batch : list
        A list containing the following elements:
        - batch[0]: Tuple of (views_1, views_2), where each is a tensor of image patch views
        - batch[2]: List of slide names for each patch
        - batch[3]: Tuple of (rows, cols), coordinates for each patch

    Returns:
    --------
    tuple
        A tuple containing two tensors:
        - views_1: The original set of image patch views
        - views_2: The set of closest neighbor views for each patch in views_1

    Notes:
    ------
    - The function uses Euclidean distance to determine the closest patches.
    - Patches from different slides are not considered as potential neighbors.
    - If a patch has no neighbors from the same slide, it is paired with itself.
    """
    # batch: ['views', 'img_path', 'slide_name', 'coord']
    views_1, views_2 = batch[0]
    slide_names = batch[2]
    rows, cols = batch[3]
    coords = torch.stack([rows, cols], dim=1).float()
    batch_size = len(slide_names)

    same_slide_matrix = torch.tensor([i==j for i in slide_names for j in slide_names]).reshape((batch_size,batch_size))
    euclidian_distance_matrix = torch.cdist(coords, coords)

    # Patches from different slides cannot be neighbors
    euclidian_distance_matrix[~same_slide_matrix] = float('inf')
    topk_values, topk_indices = torch.topk(euclidian_distance_matrix, k=2, largest=False)

    # Get the indices of those patches from different slides
    different_slide_indices = torch.where(topk_values[:,1] == float('inf'))

    # Select the indices of the neighbor patches
    selected_indices = topk_indices[:,1]

    # If one patch does not have neighbor, such patch will be its own neighbor
    selected_indices[different_slide_indices] = topk_indices[:,0][different_slide_indices]

    views_2 = views_2[selected_indices]
    return views_1, views_2
