import torch
import torch.nn as nn
import torchvision.models as pretrained_model
# First use Tokenizer (in training loopo) and than input those tokens into the models
from transformers import GPT2Model,BertModel




class ModifiedResNet(nn.Module):
    def __init__(self):
        super(ModifiedResNet, self).__init__()
        pretrained_resnet = pretrained_model.resnet50(pretrained=True)

        for param in pretrained_resnet.parameters():
            param.requires_grad = False

        self.pretrained_resnet = pretrained_resnet
        #This removes the last fully connected layer
        self.pretrained_resnet = nn.Sequential(*list(pretrained_resnet.children())[:-1])

        #Add new layer that is traineable to learn similarities of the embeddings
        self.new_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(pretrained_resnet.fc.in_features, 512),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(512, 248)
        )

    def forward(self, x):
        x = self.pretrained_resnet(x)
        x = self.new_layers(x)
        return x 
    
class ModifiedBert(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', output_dim=248):
        super(ModifiedBert, self).__init__()
        # Load the pre-trained BERT model
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        

        # Freeze the pre-trained BERT weights
        for param in self.bert.parameters():
            param.requires_grad = False

        # BERT's pooling layer output size is the same as its hidden state size
        hidden_size = self.bert.config.hidden_size
        
        # This replaces the pooler layer with a new trainable layer to match the ResNet output dimension
        # Assuming we want to use the [CLS] token embedding for classification or matching tasks
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, input_ids, attention_mask=None):
        # Pass input through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the pooled output: a summary of the content according to BERT
        # The pooled output is typically obtained by applying the BertPooler on the last layer hidden state of the [CLS] token
        pooled_output = outputs.pooler_output
        
        # Pass the pooled output through the new fully connected layer
        return self.fc(pooled_output)

#TODO: This is with GPT2 have the think anbout it as we cant as easily get the the cls token as with the other

class ModifiedTextTransformer(nn.Module):
    def __init__(self, text_transformer):
        super(ModifiedTextTransformer, self).__init__()
        self.text_transformer = text_transformer
        self.text_transformer.fc = nn.Linear(768, 10)

    def forward(self, x):
        return self.text_transformer(x)


#TODO: Combine the the tow Embedding models I guess only to a numerical value => this can than be trained with a Triplet/Contractive Loss,
class FoodClip(nn.Module):
    def __init__(self):
        super(FoodClip, self).__init__()
        self.image_embedding = ModifiedResNet()
        self.text_embedding = ModifiedBert()

    def forward(self, x):
        images, text_inputs = x
        image_embeddings = self.image_embedding(images)
        # Assuming text_inputs is a dictionary with 'input_ids' and 'attention_mask'
        text_embeddings = self.text_embedding(**text_inputs)

        # Now you have embeddings for both images and text
        # You can return them directly or process them further depending on your task
        return image_embeddings, text_embeddings