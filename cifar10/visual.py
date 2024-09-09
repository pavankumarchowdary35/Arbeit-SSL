import numpy as np
import matplotlib.pyplot as plt

# Load the data
natural_acc = np.load('metrics_M_SOTA_CIFAR10/4000/trades_attack_wrn_seed_46/4000_Natural_accuracy_epoch_100.npy')
robust_acc = np.load('metrics_M_SOTA_CIFAR10/4000/trades_attack_wrn_seed_46/4000_Robust_accuracy_epoch_100.npy')
loss = np.load('metrics_M_SOTA_CIFAR10/4000/trades_attack_wrn_seed_46/4000_LOSS_epoch_train_epoch_100.npy')

# Create an array for the epochs
epochs = np.arange(1, len(loss) + 1)

# # Plot the natural and robust accuracy
plt.plot(epochs, natural_acc, color='blue', label='Natural Accuracy')
plt.plot(epochs, robust_acc, color='orange', label='Robust Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Clean & Robust Accuracy')
plt.legend()  # Add legend to indicate the colors
plt.savefig('metrics_M_SOTA_CIFAR10/4000/trades_attack_wrn_seed_46/4000_accuracy_epoch_train_epoch_100.png')
plt.show()


# plt.plot(epochs, loss, color='green', label='Training Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training Loss over Epochs')
# plt.legend()
# plt.grid(True)
# plt.savefig('metrics_M_SOTA_CIFAR10/500/trades_attack_wrn_seed_501/500_loss_report_epoch_train_epoch_105.png')
# plt.show()