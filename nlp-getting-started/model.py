
import random
import numpy as np
import time

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

from utils import flat_accuracy, format_time

class Classifier:
    """BERT model NLP classifier"""

    def __init__(self):
        self.SEED = 42
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False,
            )

        self.optimizer = AdamW(self.model.parameters(),
                               lr=2e-5,
                               eps=1e-8
                               )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model running on {self.device}")


    def train(self, train_dataloader, val_dataloader,EPOCHS):

        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        torch.cuda.manual_seed_all(self.SEED)
        
        #create the learning rate scheduler
        total_steps = len(train_dataloader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps = 0,
            num_training_steps = total_steps)
        

        #store the average loss after each epoch so we can plot afterwards
        self.loss_values = []

        for epoch_i in range(0, EPOCHS):

            print(f"\n======== EPOCH {epoch_i+1}/{EPOCHS}")
            print("Training...")

            t0 = time.time()

            #reset total loss for this epoch
            total_loss = 0

            #put model into train mode
            self.model.train()

            for step, batch in enumerate(train_dataloader):

                if step % 40 == 0 and not step == 0:
                    elapsed = format_time(time.time() - t0)
                    print(f'** Batch {step} of {len(train_dataloader)}. Elapsed time {elapsed}')

                # `batch` contains three pytorch tensors
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                #zero the gradients
                self.model.zero_grad()

                #perform forward pass
                outputs = self.model(
                    b_input_ids,
                    token_type_ids = None,
                    attention_mask = b_input_mask,
                    labels = b_labels
                )

                loss = outputs[0]
                total_loss += loss.item()
                loss.backward()

                #prevent 'exploding gradients'
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0)

                #update parameters and take a step using the computed gradient
                self.optimizer.step()

                #update learning rate
                scheduler.step()

            avg_train_loss = total_loss/len(train_dataloader)

            self.loss_values.append(avg_train_loss)

            print("")
            print(f"   Average training loss: {avg_train_loss:.2f}")
            print(f"   Training epoch took: {format_time(time.time() - t0)}")
            
            # ===========VALIDATION============
            print("")
            print("Running validation...")

            t0 = time.time()

            self.model.eval()

            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            for batch in val_dataloader:
                batch = tuple(t.to(self.device) for t in batch)

                b_input_ids, b_input_mask, b_labels = batch

                with torch.no_grad():
                outputs = model(b_input_ids, 
                                token_type_ids=None,
                                attention_mask=b_input_mask)
                
                logits = outputs[0]

                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                tmp_eval_accuracy = flat_accuracy(logits, label_ids)

                eval_accuracy += tmp_eval_accuracy

                nb_eval_steps += 1

            print(f"   Accuracy: {eval_accuracy/nb_eval_steps}")
            print(f"   Validation took {format_time(time.time() - t0)}")

        print("")
        print("Training Completed")
    
    
    
    def test(self, inputs, masks, labels):
        pass

    def make_batch_predictions(self, inputs):
        pass
