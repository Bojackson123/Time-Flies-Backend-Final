import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import AlexNet_Weights
from PIL import Image
import numpy as np


class AlexNet:
    BUFF_SIZE = 10000

    def __init__(self):
        self.features = dict()
        self.output_prob = None

        self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        # Load the pretrained AlexNet model
        self.model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).to(self.device)

        # Set the model to evaluation mode
        self.model.eval()

        # Define the image transformations
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(227),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load labels
        self.labels = np.loadtxt('alexnet/synset_words.txt', str, delimiter='\t')

        # Initialize states and features dictionaries
        self.states = {n: [] for n in range(1000)}
        self.acc_states = {n: [0.0] for n in range(1000)}
        self.features = {}
        self.hooks = []
        self._register_hook(self.model.features[3], 'conv2')  
        self._register_hook(self.model.classifier[2], 'fc7') 
        self._register_hook(self.model.features[12], 'pool5')
        print(self.model.features)  

        print("Successfully loaded classifier")

    def _register_hook(self, layer, name):
        """
        Register a forward hook to capture and store the output of a layer
        """
        handle = layer.register_forward_hook(
            lambda module, input, output: self.features.update({name: output.detach()})
        )
        self.hooks.append(handle)

    def preprocess_image(self, frame):
        """Preprocess the image for model input."""
        image = Image.fromarray(frame) if not isinstance(frame, Image.Image) else frame
        return self.transform(image).unsqueeze(0).to(self.device)

    def run(self, frame):
        """Classify the image and manage application state."""
        image = self.preprocess_image(frame).to(self.device)

        with torch.no_grad():
            image = image.to(self.device)
            outputs = self.model(image)
            self.output_prob = torch.nn.functional.softmax(outputs, dim=1)[0]  # Get probabilities

            # Get the predicted label and its probability
            max_prob, preds = torch.max(self.output_prob, dim=0)
            new_label = str(round(max_prob.item(), 2))
            if max_prob.item() > 0.1:
                label_idx = preds.item()
                new_label = self.labels[label_idx].split(',')[0][10:]
            else:
                new_label = "-"

            # Update states and accumulated states
            for n, prob in enumerate(self.output_prob):
                self.states[n].append(prob.item())
                self.acc_states[n].append(sum(self.states[n][-self.BUFF_SIZE:]))

            self.features[3] = self.output_prob.cpu().numpy()
            self.features[2] = self.features['fc7'].cpu().numpy()
            self.features[1] = self.features['pool5'].cpu().numpy().flatten()
            self.features[0] = self.features['conv2'].cpu().numpy().flatten()

        return new_label


