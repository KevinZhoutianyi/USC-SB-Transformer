import numpy as np
import matplotlib.pyplot as plt
foldername = 'logs/LSTM-training-works-withsmalldata/'

filename1 = 'trainloss'
filename2 = 'validationloss'
def moving_average(data, window_size):
    """Compute the moving average of a list or array."""
    cumsum = np.cumsum(data)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size

# Load the loss data from the npy files
train_loss = np.load(foldername+filename1+'.npy')
validation_loss = np.load(foldername+filename2+'.npy')

# Smooth the loss data using a moving average
window_size = 5  # Choose a window size that gives the desired smoothness
smooth_train_loss = moving_average(train_loss, window_size)
smooth_validation_loss = moving_average(validation_loss, window_size)

# Create the x-axis values
x_values = np.arange(len(smooth_train_loss)) * 500 + (window_size // 2) * 500  # adjust for moving average shift

# Plot the smoothed losses
plt.figure(figsize=(10, 6))
plt.plot(x_values, smooth_train_loss, label='Training Loss', color='blue')
plt.plot(x_values, smooth_validation_loss[1:], label='Validation Loss', color='red')
plt.xlabel('Data Points Trained')
plt.ylabel('Loss')
plt.title('Smoothed Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(foldername +filename1+filename2 +'_plot')
plt.show()



import numpy as np
import matplotlib.pyplot as plt
filename = 'validationacc'
def moving_average(data, window_size):
    """Compute the moving average of a list or array."""
    cumsum = np.cumsum(data)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size

# Load the accuracy data from the npy file
validation_acc = np.load(foldername+filename+'.npy')

# Smooth the accuracy data using a moving average
window_size = 5
smooth_validation_acc = moving_average(validation_acc, window_size)

# Create the x-axis values
x_values = np.arange(len(smooth_validation_acc)) * 500 + (window_size // 2) * 500  # adjust for moving average shift

# Plot the smoothed accuracy
plt.figure(figsize=(10, 6))
plt.plot(x_values, smooth_validation_acc, label='Smoothed Validation Accuracy', color='green')
plt.xlabel('Data Points Trained')
plt.ylabel('Validation Accuracy')
plt.title('Smoothed Validation Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(foldername +filename +'_plot')
plt.show()



import numpy as np
import matplotlib.pyplot as plt
filename = 'sensitivity'
def moving_average(data, window_size):
    """Compute the moving average of a list or array."""
    cumsum = np.cumsum(data)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size

# Load the sensitivity data
sensitivity_data = np.load(foldername+filename+'.npy')

# Compute the average sensitivity for each sentence (list)
average_sensitivity = np.mean(sensitivity_data, axis=1)

# Smooth the average sensitivity data using a moving average
window_size = 5
smoothed_average_sensitivity = moving_average(average_sensitivity, window_size)

# Create the x-axis values. Adjusting for the moving average shift
x_values = np.arange(len(smoothed_average_sensitivity)) * 500 + (window_size // 2) * 500

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x_values, smoothed_average_sensitivity, label='Average Sensitivity', color='purple')
plt.xlabel('Data Points Trained')
plt.ylabel('Average Sensitivity')
plt.title('Change of Smoothed Average Sensitivity with Data Points Trained')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(foldername +filename +'_plot')
plt.show()
