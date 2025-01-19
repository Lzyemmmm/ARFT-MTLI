import pandas as pd
from transformers import BertTokenizer, BertModel
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import random
from typing import Tuple
# Load the BERT tokenizer and add special tokens
tokenizer = BertTokenizer.from_pretrained('/public/zhou/model_params/bert-base-cased')
special_tokens_dict = {'additional_special_tokens': ["#", "$", "*", "&"]}
tokenizer.add_special_tokens(special_tokens_dict)
jin_id, doler_id, star_id, and_id = tokenizer.additional_special_tokens_ids

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # Choose a fixed seed value


#========================================================================Dataset class=============================================================================#

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [label for label in df['labels']]
        self.tra1_labels = [label for label in df['tra1_labels']]
        self.tra2_labels = [label for label in df['tra2_labels']]
        self.tra1_layer = [torch.tensor(label, dtype=torch.float32) for label in df['tra1_layer']]
        self.tra2_layer = [torch.tensor(label, dtype=torch.float32) for label in df['tra2_layer']]
        self.ori_texts = [tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_labels1(self, idx):
        return np.array(self.tra1_labels[idx])

    def get_batch_labels2(self, idx):
        return np.array(self.tra2_labels[idx])

    def get_batch_tra1_layer(self, idx):
        return np.array(self.tra1_layer[idx])

    def get_batch_tra2_layer(self, idx):
        return np.array(self.tra2_layer[idx])

    def get_batch_ori_texts(self, idx):
        return self.ori_texts[idx]

    def __getitem__(self, idx):
        batch_ori_texts = self.get_batch_ori_texts(idx)
        batch_y = self.get_batch_labels(idx)
        batch_y1 = self.get_batch_labels1(idx)
        batch_y2 = self.get_batch_labels2(idx)
        batch_tra1_layer = self.get_batch_tra1_layer(idx)
        batch_tra2_layer = self.get_batch_tra2_layer(idx)
        return batch_ori_texts, batch_y, batch_y1, batch_y2, batch_tra1_layer, batch_tra2_layer



#========================================================================Model=============================================================================#



# Attention mechanism
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.scale = hidden_size ** 0.5
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        query = query.unsqueeze(1)  # (batch_size, 1, hidden_size)
        key = key.unsqueeze(2)      # (batch_size, hidden_size, 1)
        scores = torch.matmul(query, key) / self.scale  # (batch_size, 1, 1)
        scores = self.softmax(scores)  # (batch_size, 1, 1)
        attended = torch.matmul(scores, value.unsqueeze(1))  # (batch_size, 1, hidden_size)
        attended = attended.squeeze(1)  # (batch_size, hidden_size)
        return attended

# Function to remove special tokens
def remove_special_token(input_ids: torch.Tensor, attention_mask: torch.Tensor, special_token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Remove special tokens from input_ids and update attention_mask accordingly.
    """
    batch_size, seq_length = input_ids.size()
    mask = input_ids != special_token_id
    new_input_ids = torch.zeros_like(input_ids)
    new_attention_mask = torch.zeros_like(attention_mask)
    for i in range(batch_size):
        valid_tokens = input_ids[i][mask[i]]
        valid_mask = attention_mask[i][mask[i]]
        num_valid = valid_tokens.size(0)
        padding_length = seq_length - num_valid
        if num_valid > seq_length:
            valid_tokens = valid_tokens[:seq_length]
            valid_mask = valid_mask[:seq_length]
            padding_length = 0
        new_input_ids[i, :num_valid] = valid_tokens
        new_attention_mask[i, :num_valid] = valid_mask
    return new_input_ids, new_attention_mask

# FALCON model
class FALCON(nn.Module):
    def __init__(self, dropout=0.2):
        super(FALCON, self).__init__()
        self.bert = BertModel.from_pretrained('/public/zhou/model_params/bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.name_dense = nn.Linear(768, 768)
        self.time_dense = nn.Linear(768, 768)
        self.CLS_dense = nn.Linear(768, 768)
        self.place_dense = nn.Linear(768, 768)
        self.all_dense = nn.Linear(768 * 7, 768)
        self.all_dense_nt = nn.Linear(768 * 5, 768)
        self.all_dense1 = nn.Linear(768 * 4, 768)
        self.query_dense = nn.Linear(768 * 5, 768)
        self.all_dense2 = nn.Linear(768, 2)
        self.all_dense3 = nn.Linear(768, 2)
        self.entity_attention_1 = nn.Linear(768, 1)
        self.gate_tra = nn.Linear(768, 768)
        self.sigmoid = nn.Sigmoid()
        self.attention_tra = Attention(hidden_size=768)
        self.relu = nn.ReLU()

    def get_feature(self, bert_result, pool_result, x_mark_index_all, task_id, batch_size):
        """
        Extract features for different entities and tasks.
        """
        jin_result, doler_result, star_result, and_result = [], [], [], []
        for i in range(batch_size):
            jin = x_mark_index_all[i][0]
            doler = x_mark_index_all[i][1]
            star = x_mark_index_all[i][2]
            and_ = x_mark_index_all[i][3]

            if len(jin) % 2 != 0:
                jin = jin[:-1]
            if len(doler) % 2 != 0:
                doler = doler[:-1]
            if len(star) % 2 != 0:
                star = star[:-1]
            if len(and_) % 2 != 0:
                and_ = and_[:-1]

            if task_id == 1 or task_id == 2:
                name1_list = []
                for j in range(0, len(jin), 2):
                    entity = torch.mean(bert_result[i, jin[j] + 1: jin[j + 1]], dim=0, keepdim=True)
                    name1_list.append(entity)
                if len(name1_list) == 1:
                    entity1 = name1_list[0]
                else:
                    name1_tensor = torch.cat(name1_list, dim=0)
                    attention_scores = self.entity_attention_1(name1_tensor)
                    attention_weights = F.softmax(attention_scores, dim=0)
                    entity1 = torch.sum(name1_tensor * attention_weights, dim=0, keepdim=True)
                jin_result.append(entity1)

            if task_id == 1 or task_id == 3:
                name2_list = []
                for j in range(0, len(doler), 2):
                    entity = torch.mean(bert_result[i, doler[j] + 1: doler[j + 1]], dim=0, keepdim=True)
                    name2_list.append(entity)
                if len(name2_list) == 1:
                    entity2 = name2_list[0]
                else:
                    name2_tensor = torch.cat(name2_list, dim=0)
                    attention_scores = self.entity_attention_1(name2_tensor)
                    attention_weights = F.softmax(attention_scores, dim=0)
                    entity2 = torch.sum(name2_tensor * attention_weights, dim=0, keepdim=True)
                doler_result.append(entity2)

            time_list = []
            for j in range(0, len(star), 2):
                entity = torch.mean(bert_result[i, star[j] + 1: star[j + 1]], dim=0, keepdim=True)
                time_list.append(entity)
            if len(time_list) == 1:
                entity3 = time_list[0]
            else:
                time_tensor = torch.cat(time_list, dim=0)
                attention_scores = self.entity_attention_1(time_tensor)
                attention_weights = F.softmax(attention_scores, dim=0)
                entity3 = torch.sum(time_tensor * attention_weights, dim=0, keepdim=True)
            star_result.append(entity3)

            place_list = []
            for j in range(0, len(and_), 2):
                entity = torch.mean(bert_result[i, and_[j] + 1: and_[j + 1]], dim=0, keepdim=True)
                place_list.append(entity)
            if len(place_list) == 1:
                entity4 = place_list[0]
            else:
                place_tensor = torch.cat(place_list, dim=0)
                attention_scores = self.entity_attention_1(place_tensor)
                attention_weights = F.softmax(attention_scores, dim=0)
                entity4 = torch.sum(place_tensor * attention_weights, dim=0, keepdim=True)
            and_result.append(entity4)

        if task_id == 1:
            H_jin = torch.cat(jin_result, 0)
            H_doler = torch.cat(doler_result, 0)
            H_star = torch.cat(star_result, 0)
            H_and = torch.cat(and_result, 0)
            cls_dense = self.CLS_dense(self.dropout(torch.tanh(pool_result)))
            jin_dense = self.name_dense(self.dropout(torch.tanh(H_jin)))
            doler_dense = self.name_dense(self.dropout(torch.tanh(H_doler)))
            star_dense = self.time_dense(self.dropout(torch.tanh(H_star)))
            and_dense = self.place_dense(self.dropout(torch.tanh(H_and)))
            cat_result = torch.cat((cls_dense, jin_dense, doler_dense, star_dense, and_dense), 1)
            return cat_result

        if task_id == 2:
            H_jin = torch.cat(jin_result, 0)
            H_star = torch.cat(star_result, 0)
            H_and = torch.cat(and_result, 0)
            cls_dense = self.CLS_dense(self.dropout(torch.tanh(pool_result)))
            jin_dense = self.name_dense(self.dropout(torch.tanh(H_jin)))
            star_dense = self.time_dense(self.dropout(torch.tanh(H_star)))
            and_dense = self.place_dense(self.dropout(torch.tanh(H_and)))
            cat_result = torch.cat((cls_dense, jin_dense, star_dense, and_dense), 1)
            return cat_result

        if task_id == 3:
            H_doler = torch.cat(doler_result, 0)
            H_star = torch.cat(star_result, 0)
            H_and = torch.cat(and_result, 0)
            cls_dense = self.CLS_dense(self.dropout(torch.tanh(pool_result)))
            doler_dense = self.name_dense(self.dropout(torch.tanh(H_doler)))
            star_dense = self.time_dense(self.dropout(torch.tanh(H_star)))
            and_dense = self.place_dense(self.dropout(torch.tanh(H_and)))
            cat_result = torch.cat((cls_dense, doler_dense, star_dense, and_dense), 1)
            return cat_result

    def forward(self, input_id, mask, x_mark_index_all, tra1_layer, tra2_layer):
        """
        Forward pass of the FALCON model.
        """
        tra1_input_ids, tra1_attention_mask = remove_special_token(input_id, mask, doler_id)
        tra2_input_ids, tra2_attention_mask = remove_special_token(input_id, mask, jin_id)

        bert_result, pool_result = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        tra1_bert_result, tra1_pool_result = self.bert(input_ids=tra1_input_ids, attention_mask=tra1_attention_mask, return_dict=False)
        tra2_bert_result, tra2_pool_result = self.bert(input_ids=tra2_input_ids, attention_mask=tra2_attention_mask, return_dict=False)

        batch_size = input_id.size()[0]
        main_result = self.get_feature(bert_result, pool_result, x_mark_index_all, 1, batch_size)
        tra1_result = self.get_feature(tra1_bert_result, tra1_pool_result, x_mark_index_all, 2, batch_size)
        tra2_result = self.get_feature(tra2_bert_result, tra2_pool_result, x_mark_index_all, 3, batch_size)

      
        gate_tra1 = self.sigmoid(self.gate_tra(tra1_layer))
        gate_tra2 = self.sigmoid(self.gate_tra(tra2_layer))
        gated_tra1 = gate_tra1 * tra1_layer
        gated_tra2 = gate_tra2 * tra2_layer
        projected_main = self.query_dense(main_result)
        attended_tra1 = self.attention_tra(projected_main, gated_tra1, gated_tra1)
        attended_tra2 = self.attention_tra(projected_main, gated_tra2, gated_tra2)
        main_result = torch.cat((main_result, attended_tra1, attended_tra2), 1)
        final_layer = self.all_dense2(F.tanh(self.all_dense(main_result)))

        final1_layer = self.all_dense3(F.tanh(self.all_dense1(tra1_result)))
        final2_layer = self.all_dense3(F.tanh(self.all_dense1(tra2_result)))

        return final_layer, final1_layer, final2_layer
    
#========================================================================训练函数=============================================================================#




def train(model, train_data, val_data, learning_rate, epochs, batch_size, save_path, use_adaptive_weights=True):
    """
    Train a BERT classifier.
    
    Parameters:
    - model: The model to be trained.
    - train_data: Training data.
    - val_data: Validation data.
    - learning_rate: Learning rate for the optimizer.
    - epochs: Number of training epochs.
    - batch_size: Batch size for the DataLoader.
    - save_path: Path to save the best model.
    - use_adaptive_weights: Whether to use adaptive loss weights.
    """
    # Get training and validation datasets
    train_dataset = Dataset(train_data)
    val_dataset = Dataset(val_data)
    
    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Device selection
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Move model and loss function to the device
    model = model.to(device)
    criterion = criterion.to(device)
    
    # Define loss weights
    if use_adaptive_weights:
        # Use adaptive loss weights
        loss_weights = nn.Parameter(torch.ones(2, device=device), requires_grad=True)  # Corresponding to c1 and c2
        optimizer = Adam(list(model.parameters()) + [loss_weights], lr=learning_rate)
    else:
        # Use fixed loss weights
        loss_weights = torch.ones(2, device=device)  # Fixed to 1.0
        optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Initialize best accuracy and F1 score
    best_val_acc = 0.0
    best_f1_val = 0.0
    best_recall_val = 0.0
    best_precision_val = 0.0
    
    # Training loop
    for epoch_num in range(epochs):
        # Initialize training metrics
        total_acc_train = 0
        total_loss_train = 0
        true_positives_train = 0
        false_positives_train = 0
        actual_positives_train = 0
        
        model.train()
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch_num+1}/{epochs} - Training"):
            train_ori_input, train_label, train_label2, train_label3, train_tra1_layer, train_tra2_layer = batch
            x_mark_index_all = []
            input_ids = train_ori_input['input_ids'].squeeze(1).to(device)
            attention_mask = train_ori_input['attention_mask'].squeeze(1).to(device)
            
            # Process input to find indices of special tokens
            for temp in input_ids:
                temp_cup = list(enumerate(temp))
                jin_index = [index for index, value in temp_cup if value == jin_id]
                doler_index = [index for index, value in temp_cup if value == doler_id]
                star_index = [index for index, value in temp_cup if value == star_id]
                and_index = [index for index, value in temp_cup if value == and_id]
                x_mark_index = [jin_index, doler_index, star_index, and_index]
                x_mark_index_all.append(x_mark_index)
            
            # Move labels and additional layers to the device
            train_label = train_label.to(device)
            train_label2 = train_label2.to(device)
            train_label3 = train_label3.to(device)
            train_tra1_layer = train_tra1_layer.to(device)
            train_tra2_layer = train_tra2_layer.to(device)
            
            # Forward pass
            final_layer, final1_layer, final2_layer = model(input_ids, attention_mask, x_mark_index_all, train_tra1_layer, train_tra2_layer)
            
            # Compute losses for each task
            loss1 = criterion(final_layer, train_label)    # Loss for the first task
            loss2 = criterion(final1_layer, train_label2)  # Loss for the second task
            loss3 = criterion(final2_layer, train_label3)  # Loss for the third task
            
            if use_adaptive_weights:
                # Use adaptive loss weights
                c1 = torch.nn.functional.softplus(loss_weights[0])
                c2 = torch.nn.functional.softplus(loss_weights[1])
                weighted_loss = (1 / (2 * c1**2)) * loss1 + torch.log(1 + c1**2) + (1 / (2 * c2**2)) * (loss2 + loss3) / 2 + torch.log(1 + c2**2)
            else:
                # Use fixed loss weights (set to 1.0)
                weighted_loss = 0.5 * loss1 + 0.25 * loss2 + 0.25 * loss3  # Adjust weights as needed
            
            # Accumulate training loss and accuracy
            total_loss_train += loss1.item()  # Accumulate loss for the first task
            acc = (final_layer.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc
            
            # Compute true positives, false positives, and actual positives
            preds = final_layer.argmax(dim=1)
            true_positives_train += ((preds == 1) & (train_label == 1)).sum().item()
            false_positives_train += ((preds == 1) & (train_label == 0)).sum().item()
            actual_positives_train += (train_label == 1).sum().item()
            
            # Backward pass and optimization
            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()
        
        # Compute training recall, precision, and F1 score
        recall_train = true_positives_train / actual_positives_train if actual_positives_train > 0 else 0.0
        precision_train = true_positives_train / (true_positives_train + false_positives_train) if (true_positives_train + false_positives_train) > 0 else 0.0
        f1_score_train = 2 * (precision_train * recall_train) / (precision_train + recall_train) if (precision_train + recall_train) > 0 else 0.0
        
        # Initialize validation metrics
        total_acc_val = 0
        total_loss_val = 0
        true_positives_val = 0
        false_positives_val = 0
        actual_positives_val = 0
        
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch_num+1}/{epochs} - Validation"):
                val_ori_input, val_label, _, _, val_tra1_layer, val_tra2_layer = batch
                x_mark_index_all = []
                input_ids = val_ori_input['input_ids'].squeeze(1).to(device)
                attention_mask = val_ori_input['attention_mask'].squeeze(1).to(device)
                
                # Process input to find indices of special tokens
                for temp in input_ids:
                    temp_cup = list(enumerate(temp))
                    jin_index = [index for index, value in temp_cup if value == jin_id]
                    doler_index = [index for index, value in temp_cup if value == doler_id]
                    star_index = [index for index, value in temp_cup if value == star_id]
                    and_index = [index for index, value in temp_cup if value == and_id]
                    x_mark_index = [jin_index, doler_index, star_index, and_index]
                    x_mark_index_all.append(x_mark_index)
                
                # Move labels and additional layers to the device
                val_label = val_label.to(device)
                val_tra1_layer = val_tra1_layer.to(device)
                val_tra2_layer = val_tra2_layer.to(device)
                
                # Forward pass
                output, _, _ = model(input_ids, attention_mask, x_mark_index_all, val_tra1_layer, val_tra2_layer)
                
                # Compute loss
                loss_val1 = criterion(output, val_label)  # Loss for the first task
                total_loss_val += loss_val1.item()
                
                # Compute accuracy
                acc_val = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc_val
                
                # Compute true positives, false positives, and actual positives
                preds_val = output.argmax(dim=1)
                true_positives_val += ((preds_val == 1) & (val_label == 1)).sum().item()
                false_positives_val += ((preds_val == 1) & (val_label == 0)).sum().item()
                actual_positives_val += (val_label == 1).sum().item()
        
        # Compute validation recall, precision, and F1 score
        recall_val = true_positives_val / actual_positives_val if actual_positives_val > 0 else 0.0
        precision_val = true_positives_val / (true_positives_val + false_positives_val) if (true_positives_val + false_positives_val) > 0 else 0.0
        f1_score_val = 2 * (precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0.0
        val_acc = total_acc_val / len(val_dataset)
        
        # Output results
        if use_adaptive_weights:
            current_c1 = torch.nn.functional.softplus(loss_weights[0]).item()
            current_c2 = torch.nn.functional.softplus(loss_weights[1]).item()
        else:
            current_c1, current_c2 = 1.0, 1.0  # Fixed weights
        
        print(
            f'''Epoch {epoch_num + 1}/{epochs}
            | Train Loss: {total_loss_train / len(train_dataloader):.3f} 
            | Train Accuracy: {total_acc_train / len(train_dataset):.3f} 
            | Train Recall: {recall_train:.3f}
            | Train Precision: {precision_train:.3f}
            | Train F1 Score: {f1_score_train:.3f}
            | Val Loss: {total_loss_val / len(val_dataloader):.3f} 
            | Val Accuracy: {val_acc:.3f} 
            | Val Recall: {recall_val:.3f}
            | Val Precision: {precision_val:.3f}
            | Val F1 Score: {f1_score_val:.3f}
            | Loss Weights: c1={current_c1:.4f}, c2={current_c2:.4f}'''
        )
        
        # Save the model with the highest F1 score on the validation set
        if f1_score_val > best_f1_val:
            best_val_acc = val_acc
            best_f1_val = f1_score_val
            best_recall_val = recall_val
            best_precision_val = precision_val
            torch.save(model.state_dict(), save_path)
    
    # Return the best validation metrics and loss weights
    return {
        'best_val_acc': best_val_acc,
        'best_f1_val': best_f1_val,
        'best_recall_val': best_recall_val,
        'best_precision_val': best_precision_val
    }



if __name__ == '__main__':

    # 载入数据
    train_data = pd.read_pickle('sample_train_data.pkl')
    val_data = pd.read_pickle('sample_val_data.pkl')

    # 指定参数
    model=FALCON()
    epochs=50
    batch_size=16
    learning_rate = 5e-6
    save_path='your_path'
    use_adaptive_weights=True

    # 开始训练
    train(model, train_data, val_data, learning_rate, epochs,batch_size, save_path, use_adaptive_weights=use_adaptive_weights)