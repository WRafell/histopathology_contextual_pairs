import torch
import numpy as np


def two_distortion_positive_pairs(batch: list) -> tuple:
    views_1, views_2 = batch[0][:2]
    return views_1, views_2


def three_views_positive_pairs(batch: list) -> tuple:
    pivot_view_1, pivot_view_2, neighbor_view = batch[0][:3]
    return pivot_view_1, pivot_view_2, neighbor_view


class ContextPositivePairs():
    def __init__(self, minimum_distance: int = None, normal_label: int = None):
        self.minimum_distance = minimum_distance
        self.normal_label = normal_label


    def closest_patch_positive_pairs(self, batch: list) -> tuple:

        assert self.minimum_distance is not None

        # batch: ['views', 'img_path', 'slide_name', 'coord']
        views_1, views_2 = batch[0][:2]
        slide_names = batch[2]
        coords = batch[3]
        coords = torch.stack(coords, dim=1).float()
        batch_size = len(slide_names)

        # change to diff_slide_matrix
        same_slide_matrix = torch.tensor([i==j for i in slide_names for j in slide_names]).reshape((batch_size,batch_size))
        
        # Chebyshev distance
        relative_distance_matrix = torch.cdist(coords, coords, p=float('inf'))

        # Patches from different slides cannot be neighbors
        relative_distance_matrix[~same_slide_matrix] = float('inf')

        if self.minimum_distance > 0:
            relative_distance_matrix[relative_distance_matrix > self.minimum_distance] = float('inf')

        topk_values, topk_indices = torch.topk(relative_distance_matrix, k=2, largest=False)

        # Get the indices of those patches from different slides
        different_slide_indices = torch.where(topk_values[:,1] == float('inf'))

        # Select the indices of the neighbor patches
        selected_indices = topk_indices[:,1]

        # If one patch does not have neighbor, such patch will be its own neighbor
        selected_indices[different_slide_indices] = topk_indices[:,0][different_slide_indices]

        views_2 = views_2[selected_indices]
        return views_1, views_2
    

    def random_patch_positive_pairs(self, batch: list) -> tuple:
        """
        Randomly selects a positive patch pair from the same slide for a batch of image patches.

        This function takes a batch of image patches and their associated metadata, and for each patch,
        it selects a random neighboring patch from the same slide. If a patch has no neighbors from
        the same slide, it is paired with itself.

        Parameters:
        -----------
        batch : list
            A list containing the following elements:
            - batch[0][:2]: Tuple of (views_1, views_2), where each is a tensor of image patch views
            - batch[2]: List of slide names for each patch
            - batch[3]: Tuple of (rows, cols), coordinates for each patch

        Returns:
        --------
        tuple
            A tuple containing two tensors:
            - views_1: The original set of image patch views
            - views_2: The set of randomly selected neighbor views for each patch in views_1

        Notes:
        ------
        - Patches from different slides are not considered as potential neighbors.
        - If a patch has no neighbors from the same slide, it is paired with itself.
        """
        # batch: ['views', 'img_path', 'slide_name', 'coord']
        views_1, views_2 = batch[0][:2]
        slide_names = batch[2]
        batch_size = len(slide_names)

        # Create a matrix to identify patches from the same slide
        same_slide_matrix = torch.tensor(
            [i==j for i in slide_names for j in slide_names]
        ).reshape((batch_size,batch_size))

        # Randomly select a neighbor for each patch within the same slide
        selected_indices = []
        for i in range(batch_size):
            same_slide_indices = torch.where(same_slide_matrix[i])[0]
            if len(same_slide_indices) > 1:
                # Exclude self from random selection
                same_slide_indices = same_slide_indices[same_slide_indices != i]
            if len(same_slide_indices) > 0:
                # Randomly select one neighbor
                selected_index = same_slide_indices[torch.randint(len(same_slide_indices), (1,))].item()
            else:
                # If no neighbors are available, pair the patch with itself
                selected_index = i
            selected_indices.append(selected_index)

        selected_indices = torch.tensor(selected_indices, dtype=torch.long)

        # Select the views_2 patches based on the randomly chosen indices
        views_2 = views_2[selected_indices]
        return views_1, views_2
    

    def diff_normal_slide_positive_pairs(self, batch: list) -> tuple:

        assert self.normal_label is not None

        views_1, views_2 = batch[0][:2]
        slide_names = batch[2]
        slide_labels = batch[4]
        batch_size = len(slide_labels)
        device = slide_labels.device

        # Identify patches with the label `normal_label`
        normal_mask = slide_labels == self.normal_label
        normal_indices = torch.where(normal_mask)[0]

        # Create a pairwise comparison matrix for labels and slides
        same_label = slide_labels.unsqueeze(1) == slide_labels.unsqueeze(0)
        diff_slide = torch.tensor(
            [i!=j for i in slide_names for j in slide_names], device=device
        ).reshape((batch_size,batch_size))

        valid_pairs = same_label & diff_slide & normal_mask.unsqueeze(1)

        # Default selected indices: pair each patch with itself
        selected_indices = torch.arange(batch_size, device=device)

        for i in normal_indices:
            valid_indices = torch.where(valid_pairs[i])[0]
            if valid_indices.numel() > 0:
                # Randomly select one valid index
                selected_indices[i] = valid_indices[torch.randint(len(valid_indices), (1,), device=device)]

        # Select the views_2 patches based on the selected indices
        views_2 = views_2[selected_indices]

        return views_1, views_2
    

    def diff_slide_same_label_positive_pairs(self, batch: list) -> tuple:

        views_1, views_2 = batch[0][:2]
        slide_names = batch[2]
        slide_labels = batch[4]
        batch_size = len(slide_labels)
        device = slide_labels.device

        # Create a pairwise comparison matrix for labels and slides
        same_label = slide_labels.unsqueeze(1) == slide_labels.unsqueeze(0)
        diff_slide = torch.tensor(
            [i!=j for i in slide_names for j in slide_names], device=device
        ).reshape((batch_size,batch_size))

        valid_pairs = same_label & diff_slide

        # Default selected indices: pair each patch with itself
        selected_indices = torch.arange(batch_size, device=device)

        # Iterate over all patches and select a valid pair
        for i in range(batch_size):
            valid_indices = torch.where(valid_pairs[i])[0]
            if valid_indices.numel() > 0:
                # Randomly select one valid index
                selected_indices[i] = valid_indices[torch.randint(len(valid_indices), (1,), device=device)]

        # Select the views_2 patches based on the selected indices
        views_2 = views_2[selected_indices]

        return views_1, views_2
