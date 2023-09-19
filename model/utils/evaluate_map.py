import torch

def generate_mrhisum_seg_scores(cp_frame_scores, uniform_clip=5):
    # split in uniform division
    splits = torch.split(cp_frame_scores, uniform_clip)
    averages = [torch.mean(torch.unsqueeze(sp, 0), dim=1) for sp in splits]

    segment_scores = torch.cat(averages)
    
    return segment_scores

def top50_summary(scores):
    sort_idx = torch.argsort(scores, descending=True)
    # take the 50% shots
    median_index = len(scores) // 2 
    filtered_sort_idx = sort_idx[:median_index]
    selected_segs = [0] * len(scores)
    for index in filtered_sort_idx:
        selected_segs[index] = 1
    
    return selected_segs

def top15_summary(scores):
    sort_idx = torch.argsort(scores, descending=True)

    # take the 15% shots
    filter_index = int(len(scores) * 0.15) 
    filtered_sort_idx = sort_idx[:filter_index]
    selected_segs = [0] * len(scores)
    for index in filtered_sort_idx:
        selected_segs[index] = 1
    
    return selected_segs