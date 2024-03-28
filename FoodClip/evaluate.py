import numpy as np
import faiss
import torch
from torch.utils.data import DataLoader
from train import create_train_val_test_with_Data_Loader
from model import FoodClip
import os,json
from tqdm import tqdm

def evaluate(model, device, test_dataloader):

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(f"{root_dir}/", 'FoodClip/models/model_FoodClip.pth')  # requires `import os`
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Example for images
    image_embeddings = []
    text_embeddings = []
    image_paths = []
    texts = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Validating..."):
            images_true = batch['image_true'].to(device)
            texts_true = {
            'input_ids': batch['text_true']['input_ids'].to(device),
            'attention_mask': batch['text_true']['attention_mask'].to(device)
            }
            image_paths.extend([path for path in batch['image_path']])
            texts.extend([text for text in batch['text_clear']])

            # Forward pass for each part of the triplet
            true_embedding_image, true_embedding_text = model((images_true, texts_true))
            image_embeddings.append(true_embedding_image)
            text_embeddings.append(true_embedding_text)

    # Concatenate all embeddings into a single tensor
    image_embeddings = torch.cat(image_embeddings, dim=0)
    text_embeddings = torch.cat(text_embeddings, dim=0)
    print(f"Image Embeddings Tensor Shape: {image_embeddings.shape}")

    # Convert embeddings to numpy arrays
    image_embeddings_np = image_embeddings.cpu().numpy()
    print(f"Image Embeddings Numpy Shape: {image_embeddings_np.shape}")
    text_embeddings_np = text_embeddings.cpu().numpy()


    #The L2 Distance works here as Triplet Loss works with the L2 Distance
    # Create a FAISS index for the embeddings (L2 distance is used by default)
    index = faiss.IndexFlatL2(text_embeddings_np.shape[1])   #this tells the dmiension of the embedding
    index.add(text_embeddings_np)  # add all the embeddings to the index

    # Loop over the image embeddings and find the closest text for each
    image_output = []
    for i in range(len(image_embeddings_np)):
        query_image_embedding = image_embeddings[i]
        query_image_embedding_np = query_image_embedding.cpu().numpy()

        # Perform the search
        D, I = index.search(query_image_embedding_np.reshape(1, -1), k=3)  # k=1 for the closest, increase for more results
        print(f"Closest Text Embedding Indices: {I}")

        # I contains the indices of the closest embeddings in the database
        closest_image_idx = I[0][0]

        # Work with image paths and embeddings in a nearest neighbor library
        image_path = image_paths[i]
        closest_text = texts[closest_image_idx]
        image_output.append({"image path": image_path, "text": closest_text})

        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        json_path = os.path.join(f"{root_dir}/", 'FoodClip/results/neareast_neighbour_images.json')  # requires `import os`
        with open(json_path, 'w') as f:
            json.dump(image_output, f, indent=4)
    return image_output


def get_images_from_textlist():
    pass
def get_text_from_imagelist():
    pass

if __name__ == "__main__":
    model = FoodClip()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader, val_dataloader, test_dataloader = create_train_val_test_with_Data_Loader()
    evaluate(model, device, test_dataloader)
            