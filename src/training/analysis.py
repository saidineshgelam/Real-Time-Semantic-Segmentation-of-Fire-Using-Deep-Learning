from typing import Dict
import matplotlib.pyplot as plt
import numpy as np

def _fig_draw(val, his, val_history,title, value_name):
    plt.subplot(*val)
    plt.title(title)

    
    plt.plot(his, label='train')
    plt.plot(val_history, label='validation')

    
    plt.xlabel('epochs')
    plt.ylabel(value_name)

    plt.legend()

def plot_training_history(history) :
  
    plt.figure(figsize=(15, 10))
    _fig_draw((2, 1, 1), history['train_loss'], history['val_loss'],
                  ' loss history', 'loss')
    _fig_draw((2, 2, 3), history['train_miou'], history['val_miou'],
                  ' MIoU history', 'MIoU')
    _fig_draw((2, 2, 4), history['train_mpa'],
                  history['val_mpa'],
                  ' MPA history', 'MPA')

    plt.suptitle('History', size=16)

    plt.tight_layout()
    plt.show()
