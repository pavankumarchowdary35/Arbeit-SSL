Semi supervised Learning with Limited labels:

• Performed semi-supervised learning with Wide-Resnet-28-5 model and dynamic pseudo-label generation for enhancing
robustness.

• Trained a teacher model on the fix-match algorithm and used that teacher model to generate Pseudo labels for the
unlabeled data.

• The pseudo labels are strategically updated at every epoch with the combination of pseudo labels generated by teacher
model, current model predictions( after the warm-up phase training) and the predictions form the previous epoch.

• Implemented FGSM and PGD adversarial training to enhance robustness against adversarial examples.

• Employed Dropout and Mix-up regularization to prevent overfitting during semi supervised training.

• Used TRADES loss function in the training to improve the robustness of the network against adversarial attacks