import os
import torch
from torch import nn

class Checkpoint(): 
    def __init__(self, position, final_accuracy: float = 0.):

        self.new_accuracy = final_accuracy
        self.position = position
        os.makedirs(os.path.dirname(self.position), exist_ok=True)

    def load_best_weights(self, model):
        checkpoint = torch.load(self.position)
        model.load_state_dict(checkpoint['model_state_dict'])

    def save_best(self, model, optimizer,new_accuracy) :
        if new_accuracy > self.new_accuracy:
            checkpoint = {}
            checkpoint['model_state_dict'] = model.state_dict()
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            checkpoint['best_accuracy'] = new_accuracy

            torch.save(checkpoint, self.position)
            self.new_accuracy = new_accuracy

