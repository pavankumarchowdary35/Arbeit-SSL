Semi-Supervised Teacher-Student Model:
Employed a teacher-student framework where a teacher model generates pseudo-labels for unlabeled data, enhancing the training of a student model with a combination of labeled and pseudo-labeled data.

Iterative Learning with Consistency:
Applied a matching criterion to focus training on images with consistent predictions across clean and adversarial examples(Consistency Regularization), gradually improving the model’s robustness.

Robust Loss Integration:
Integrated multiple loss components (Lnat, Lrob, Ladv) where Lnat and Ladv focus on accurate classification, while Lrob improves consistency between natural and adversarial examples.

Unmatched Images Utilization: 
Introduced additional loss terms based on KL Divergence to leverage unmatched images with label flips, aligning model predictions with soft labels from the teacher model to enhance overall robustness and accuracy.




## Documentation

You can view the full documentation [here](Pavan_kumar_arbeit_1633647.pdf).