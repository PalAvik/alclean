from collections import defaultdict


def create_cub_label_siblings(bird_categories):
    """
    Create a dictionary of sibling labels for each bird category in the CUB dataset.
    :param bird_categories: List of bird categories in the CUB dataset
    :return: Dictionary of sibling labels for each bird category
    """
    sibling_labels = defaultdict(list)

    for category in bird_categories:
        bird_type = category.split('_')[-1]
        sibling_labels.setdefault(category, [])

        for other_cat in bird_categories:
            if other_cat == category:
                continue
            else:
                other_bird_type = other_cat.split('_')[-1]
                if other_bird_type == bird_type:
                    sibling_labels[category].append(other_cat)
    
    return sibling_labels
