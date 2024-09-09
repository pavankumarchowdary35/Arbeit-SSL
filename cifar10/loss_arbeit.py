def TRADES_loss_main_training(model, x_natural, y, soft_labels, attack, index):
    # Generate adversarial examples
    x_adv, _ = attack.perturb(x_natural, y)
    
    # labeled_indexes = np.load('checkpoint_paper/sampled_label_idx_4000.npy')
    labeled_indexes = np.load('checkpoint_paper/500/sampled_label_idx_500 (4).npy')
    labeled_indexes = torch.tensor(labeled_indexes, dtype=torch.long)
    
    # Ensure labeled_indexes is on the same device as x_natural
    device = x_natural.device
    labeled_indexes = labeled_indexes.to(device)
    
    # Calculate logits for natural and adversarial examples
    logits_natural = model(x_natural)
    logits_adv = model(x_adv)
    
    # Get predicted labels for natural and adversarial examples
    preds_natural = logits_natural.argmax(dim=1)
    preds_adv = logits_adv.argmax(dim=1)
    
    # Get teacher model's predicted labels
    teacher_preds = y
    # teacher_preds = soft_labels.argmax(dim=1)
    
    # Find the indices where the predictions match with teacher labels
    matching_indices = (preds_natural == preds_adv) & (preds_natural == teacher_preds)
    unmatched_indices = ~matching_indices
    
    matching_indices = index[matching_indices].tolist()  
    unmatched_indices = index[unmatched_indices].tolist()
    # print('matching indexes are', len(matching_indices))  
    # print('indexes are', index)
    batch_labeled_indexes = [idx for idx in labeled_indexes.tolist() if idx in index]
    # print('labeled images are', len(batch_labeled_indexes))

    # Combine matching indices with labeled indexes
    combined_indices = list(set(matching_indices + batch_labeled_indexes))
    # print('combined indices are', len(combined_indices))
    
    index_to_batch_idx = {idx: i for i, idx in enumerate(index.tolist())}    
    batch_combined_indices = [index_to_batch_idx[idx] for idx in combined_indices if idx in index_to_batch_idx]
    # print(batch_combined_indices)
    
    # Filter natural and adversarial examples to only include combined indices
    x_natural_matching = x_natural[batch_combined_indices]
    y_matching = y[batch_combined_indices]           
    x_adv_matching = x_adv[batch_combined_indices]
    
    # Calculate logits for combined indices
    logits_natural_matching = logits_natural[batch_combined_indices]
    logits_adv_matching = logits_adv[batch_combined_indices]
    
    # Calculate probabilities for all examples
    nat_probs_ul_all = F.softmax(logits_natural, dim=1)
    adv_probs_ul_all = F.softmax(logits_adv, dim=1)
    
    # Calculate probabilities for combined indices
    nat_probs_ul = nat_probs_ul_all[batch_combined_indices]
    adv_probs_ul = adv_probs_ul_all[batch_combined_indices]
       
    batch_unmatched_indices = [index_to_batch_idx[idx] for idx in unmatched_indices if idx in index_to_batch_idx]
    nat_probs_unmatched = nat_probs_ul_all[batch_unmatched_indices]
    adv_probs_unmatched = adv_probs_ul_all[batch_unmatched_indices]
    
    soft_labels_unmatched = soft_labels[batch_unmatched_indices]
    
    # Natural loss with true labels for combined indices
    loss_natural = F.cross_entropy(logits_natural_matching, y_matching)
    
    # Robust loss with KL divergence for all examples
    loss_robust = F.kl_div((adv_probs_ul_all + 1e-12).log(), nat_probs_ul_all, reduction='batchmean')
    
    # Additional loss term: cross-entropy of adversarial examples with true labels for combined indices
    loss_adv = F.cross_entropy(logits_adv_matching, y_matching)
    loss_kl_unmatched = F.kl_div((nat_probs_unmatched + 1e-12).log(), soft_labels_unmatched, reduction='sum')
    loss_kl_adv_unmatched = F.kl_div((adv_probs_unmatched + 1e-12).log(), soft_labels_unmatched, reduction='batchmean')
                                
    # Total loss
    loss = loss_natural + 0.5 * loss_kl_unmatched + 7 * loss_robust + loss_adv + loss_kl_adv_unmatched
    
    return loss


def TRADES_loss_warmup(model,x_natural, y, soft_labels, attack, index):
    x_adv, _ = attack.perturb(x_natural,y)
    logits = model(x_natural)
    nat_probs_ul = F.softmax(logits, dim=1)
    # print('probs are', nat_probs_ul)
    # print('soft labels are' , soft_labels)
    loss_natural = F.cross_entropy(nat_probs_ul, y)
    
    adv_outputs_ul = model(x_adv)
    adv_probs_ul = F.softmax(adv_outputs_ul, dim=1)

    loss_robust = F.kl_div((adv_probs_ul+1e-12).log(), nat_probs_ul, reduction = 'batchmean')
    loss_adv = F.cross_entropy(adv_outputs_ul, y)
    # print('robust loss is ',loss_robust)
    # print('natural loss is', loss_natural)
    loss = loss_natural + 6 * loss_robust + loss_adv     
    return loss