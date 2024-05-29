import torch
from torch.utils.data import Dataset



class SequenceDataset_TaskDetector(Dataset):

    def __init__(self,X, sequence_length=10):
        """
        Initializes the SequenceDataset.
        
        Args:
            y (torch.Tensor): The target labels for each sequence.
            X (torch.Tensor): The input sequences.
            sequence_length (int): The desired length of each sequence.
        """
        self.sequence_length = sequence_length
        self.X = torch.tensor(X)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return self.X.shape[0] * self.X.shape[1]

    def __getitem__(self, i): 
        """
        Gets the i-th sample from the dataset.
        
        Args:
            i (int): Index of the desired sample.
        
        Returns:
            xx (torch.Tensor): Input sequence of length sequence_length.
            yy (torch.Tensor): Corresponding target sequence.
        """
        trial_index = i // self.X.shape[1]
        point_index = i % self.X.shape[1]
        
        if point_index > self.sequence_length - 1:
            point_start = point_index - self.sequence_length
            xx = self.X[trial_index, point_start:point_index, :]
            yy = self.X[trial_index, point_index, :]
        else:
            padding_x = self.X[trial_index, 0:1, :].repeat(self.sequence_length - point_index, 1)
            xx = self.X[trial_index, 0:point_index, :]
            xx = torch.cat((padding_x, xx), dim=0)
            yy = self.X[trial_index, point_index, :]
            
        return xx, yy
    


class SequenceDataset(Dataset):

    def __init__(self, y, X, sequence_length=10):
        """
        Initializes the SequenceDataset.
        
        Args:
            y (torch.Tensor): The target labels for each sequence.
            X (torch.Tensor): The input sequences.
            sequence_length (int): The desired length of each sequence.
        """
        self.sequence_length = sequence_length
        self.y = torch.tensor(y)
        self.X = torch.tensor(X)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return self.X.shape[0] * self.X.shape[1]

    def __getitem__(self, i): 
        """
        Gets the i-th sample from the dataset.
        
        Args:
            i (int): Index of the desired sample.
        
        Returns:
            xx (torch.Tensor): Input sequence of length sequence_length.
            yy (torch.Tensor): Corresponding target sequence.
        """
        trial_index = i // self.X.shape[1]
        point_index = i % self.X.shape[1]
        
        if point_index > self.sequence_length - 1:
            point_start = point_index - self.sequence_length
            xx = self.X[trial_index, point_start:point_index, :]
            yy = self.y[trial_index, point_start+1:point_index+1, :]
        else:
            padding_x = self.X[trial_index, 0:1, :].repeat(self.sequence_length - point_index, 1)
            padding_y = self.y[trial_index, 0:1, :].repeat(self.sequence_length - point_index - 1, 1)
            xx = self.X[trial_index, 0:point_index, :]
            xx = torch.cat((padding_x, xx), dim=0)
            yy = self.y[trial_index, 0:point_index + 1, :]
            yy = torch.cat((padding_y, yy), dim=0)
            
        return xx, yy    