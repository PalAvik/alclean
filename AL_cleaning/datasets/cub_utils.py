from collections import defaultdict


def create_cub_label_siblings(root, bird_categories, plot_graph=False):
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
