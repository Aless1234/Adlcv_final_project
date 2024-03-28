import torch
import os
import csv
from tqdm import tqdm
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn import TripletMarginLoss, CosineEmbeddingLoss
from transformers import GPT2Tokenizer,BertTokenizer
from model import FoodClip
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate

def custom_collate_fn(batch):
    # Handling images
    images_true = [item['image_true'] for item in batch]
    images_false = [item['image_false'] for item in batch]
    # No need for special handling here as images are already resized by the dataset's transform

    # Handling text inputs
    # Padding text inputs dynamically to the max length in this batch
    text_true_input_ids = pad_sequence([item['text_true']['input_ids'].squeeze(0) for item in batch], batch_first=True)
    text_false_input_ids = pad_sequence([item['text_false']['input_ids'].squeeze(0) for item in batch], batch_first=True)

    # Handling attention masks
    text_true_attention_mask = pad_sequence([item['text_true']['attention_mask'].squeeze(0) for item in batch], batch_first=True)
    text_false_attention_mask = pad_sequence([item['text_false']['attention_mask'].squeeze(0) for item in batch], batch_first=True)

    # Reconstructing the batch
    batch = {
        'image_true': default_collate(images_true),
        'image_false': default_collate(images_false),
        'text_true': {'input_ids': text_true_input_ids, 'attention_mask': text_true_attention_mask},
        'text_false': {'input_ids': text_false_input_ids, 'attention_mask': text_false_attention_mask},
        'image_path': [item['image_path'] for item in batch],
        'text_clear': [item['text_clear'] for item in batch]
    }

    return batch


def contrastive_loss(output1, output2, label, margin=2.0):
    pass

#We return both true and false image and text, so we can run the model for both to get the embedding even though we only need true anf false text for the loss
class FoodClipDataset(Dataset):
    def __init__(self, images, texts, transform=None, tokenizer_name='bert-base-uncased'):
        """
        Args:
            images (list of str): List of image file paths.
            texts (list of str): List of texts corresponding to each image.
            transform (callable, optional): Optional transform to be applied on an image.
            tokenizer (callable, optional): Optional tokenizer function to process the text.
        """
        self.images, self.texts = self.filter_valid_images(images, texts)
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    def filter_valid_images(self, images, texts):
        valid_images, valid_texts = [], []
        for image, text in zip(images, texts):
            try:
                with Image.open(image) as img:
                    valid_images.append(image)
                    valid_texts.append(text)
            except Exception as e:
                print(f"Skipping image {image}: {e}")
        return valid_images, valid_texts
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        random_idx = idx
        while random_idx == idx:
            random_idx = np.random.randint(0, len(self.texts))

        image_path_true = self.images[idx]
        image_path_false = self.images[random_idx]
        image_true = Image.open(image_path_true).convert('RGB')  # Ensure image is in RGB format
        image_false = Image.open(image_path_false).convert('RGB')  # Ensure image is in RGB format

        if self.transform:
            image_true = self.transform(image_true)
            image_false = self.transform(image_false)
        
        text_true = self.texts[idx]
        text_false = self.texts[random_idx]

        text_input_true = self.tokenizer(text_true, return_tensors="pt", padding=True, truncation=True, max_length=512)
        text_input_false = self.tokenizer(text_false, return_tensors="pt", padding=True, truncation=True, max_length=512)

        return {'image_true': image_true, 'image_false':image_false, 'text_true': text_input_true, 'text_false': text_input_false, "image_path": image_path_true, "text_clear": text_true}


# Create an instance of your custom dataset
def create_train_val_test_with_Data_Loader():
    image_path_ordered = []
    food_title_ordered = []
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(f"{root_dir}/", 'Data/Food_Ingredients.csv')  # requires `import os`
    # Open the CSV file
    with open(csv_path, 'r') as file:
        # Create a CSV reader
        reader = csv.reader(file)
        next(reader, None)
        
        # Iterate over each row in the CSV file
        for row in reader:
            # Get the image name from the desired column (e.g., column index 0)
            image_name = row[4]
            
            # Construct the file path to the image
            image_path = os.path.join(f"{root_dir}/", f"Data/FoodImages/FoodImages/{image_name}.jpg")
            #Add path and title to the lists
            image_path_ordered.append(image_path)
            food_title_ordered.append(row[1])

    # Create an instance of your torch transform
    transform = transforms.Compose([
        transforms.Resize((169, 274)),
        transforms.ToTensor()
    ])

    images_train, images_val_test, texts_train, texts_val_test = train_test_split(image_path_ordered, food_title_ordered, test_size=0.3, random_state=42)
    images_test, images_val, texts_test, texts_val = train_test_split(images_val_test, texts_val_test, test_size=0.5, random_state=42)

    # Now, create your training and validation datasets
    train_dataset = FoodClipDataset(images_train, texts_train, transform=transform)
    val_dataset = FoodClipDataset(images_val, texts_val, transform=transform)
    test_dataset = FoodClipDataset(images_test, texts_test, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False,collate_fn=custom_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False,collate_fn=custom_collate_fn)
    return train_dataloader, val_dataloader, test_dataloader

def train(model, train_dataloader, val_dataloader, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Initialize the TripletMarginLoss
    loss_function = TripletMarginLoss(margin=0.5)
    optimizer = Adam(model.parameters(), lr=0.001)
    
    writer = SummaryWriter()

    for epoch in range(epochs):
        
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            images_true, images_false = batch['image_true'].to(device), batch['image_false'].to(device)
            text_true = {
            'input_ids': batch['text_true']['input_ids'].to(device),
            'attention_mask': batch['text_true']['attention_mask'].to(device)
            }
            text_false = {
                'input_ids': batch['text_false']['input_ids'].to(device),
                'attention_mask': batch['text_false']['attention_mask'].to(device)
            }

            # Forward pass for each part of the triplet
            true_embedding_image, true_embedding_text = model((images_true, text_true))
            false_embedding_image, false_embedding_text = model((images_false, text_false))

            
            # Compute triplet loss
            loss = loss_function(true_embedding_image, true_embedding_text, false_embedding_text)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Logging the average loss
        average_train_loss = train_loss / len(train_dataloader)
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validating..."):
                images_true, images_false = batch['image_true'].to(device), batch['image_false'].to(device)
                text_true = {
                'input_ids': batch['text_true']['input_ids'].to(device),
                'attention_mask': batch['text_true']['attention_mask'].to(device)
                }
                text_false = {
                    'input_ids': batch['text_false']['input_ids'].to(device),
                    'attention_mask': batch['text_false']['attention_mask'].to(device)
                }
                # Forward pass for each part of the triplet
                true_embedding_image, true_embedding_text = model((images_true, text_true))
                false_embedding_image, false_embedding_text = model((images_false, text_false))

                
                # Compute triplet loss
                loss = loss_function(true_embedding_image, true_embedding_text, false_embedding_text)
                    
                val_loss += loss.item()
        
        # Logging the validation loss
        model_save_counter_validation_loss_change= 0
        average_val_loss = val_loss / len(val_dataloader)
        previous_val_loss = 999999
        writer.add_scalars('Loss', {
            'Training': average_train_loss,
            'Validation': average_val_loss
        }, epoch)
        if average_val_loss > previous_val_loss:
            model_save_counter_validation_loss_change += 1
            if model_save_counter_validation_loss_change > 3:
                print(f"Validation loss has not improved for 3 epochs. Early stopping...")
                break
        else:
            model_save_counter_validation_loss_change = 0
            torch.save(model.state_dict(), f"models/model_FoodClip.pth")

        previous_val_loss = average_val_loss

    writer.close()


if __name__ == "__main__":
    model = FoodClip()
    train_dataloader, val_dataloader, test_dataloader = create_train_val_test_with_Data_Loader()
    train(model, train_dataloader, val_dataloader, epochs=30)