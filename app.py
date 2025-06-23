import streamlit as st
import torch
import matplotlib
from PIL import Image
import numpy as np

class Generator(torch.nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(z_dim, 256, 4, 1, 0),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(256, 128, 4, 2, 1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(128, 64, 4, 2, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(64, 1, 4, 2, 1),
            torch.nn.Tanh()
        )
    
    def forward(self, x): return self.net(x)

@st.cache_resource
def load_model():
    model = Generator()
    model.load_state_dict(torch.load('generator.pth', map_location='cpu'))
    model.forward
    return model

generator = load_model()

st.title("MNIST Digit Image Generator")
digit = st.slider("Select a digit (0-9):", 0, 9, 5)

fixed_z = torch.randn(5, 100, 1, 1)

labels = torch.tensor([digit] * 5)  # shape: [5]

with torch.no_grad():
    generated = generator(fixed_z).cpu().numpy()

images = []
for img in generated:
    img = (img.squeeze() + 1) / 2 * 255
    img = Image.fromarray(img.astype(np.uint8), mode='L')
    images.append(img)

st.write(f"Generated images for digit: {digit}")
cols = st.columns(5)
for col, img in zip(cols, images):
    col.image(img, use_container_width=True)