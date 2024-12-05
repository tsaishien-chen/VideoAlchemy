import numpy as np
import cv2
from typing import Dict, Tuple, List, Any, Union

import torch
import torch.nn as nn
import torch.distributed
import torch.distributed as dist

class DistributedTemporalConsistency(nn.Module):
    """
    Class for distributed Temporal Flickering score computation
    """
    def __init__(self):
        super().__init__()
        
        # Instantiates the mu and sigma tensors
        self.score_sum = torch.zeros((), dtype=torch.float64, device="cuda")
        self.accumulated_elements = 0

        # Waits that each process loads the feature extractor
        dist.barrier()

    def accumulate_stats(self, video: np.ndarray):
        """
        Accumulates the statistics
        :param video: (batch_size, frames_count, height, width, 3) numpy array with videos.
        """
        # Computes the mean absolute error (MAE) between two consecutive frames
        frame_absdiffs = []
        for i in range(video.shape[1] - 1):
            frame_absdiffs.append(
                cv2.absdiff(
                    np.array(video[:,i], dtype=np.float32),
                    np.array(video[:,i+1], dtype=np.float32)
                )
            )
            
        frame_absdiffs = np.stack(frame_absdiffs, axis=1)
        current_score_sum = sum([
            1.0 - np.mean(frame_absdiffs[i]).item() / 255
            for i in range(frame_absdiffs.shape[0])
        ])

        # Updates the total similarity
        self.score_sum = self.score_sum + current_score_sum

        # Updates the number of accumulated elements
        elements_count = video.shape[0]
        elements_count = torch.as_tensor(elements_count, dtype=torch.int64, device="cuda")
        # Uses distirbuted to get the current images from all processes
        torch.distributed.all_reduce(elements_count)
        self.accumulated_elements += int(elements_count.item())

    def get_statistics(self) -> torch.Tensor:
        """
        Performs computation of mu and sigma based on the accumulated statistics
        :return: () scalar tensor with the average cosine similarity
        """

        score_sum = self.score_sum
        torch.distributed.all_reduce(score_sum)
        
        # Computes mean and covariance estimates
        score_sum = score_sum / self.accumulated_elements

        return score_sum
