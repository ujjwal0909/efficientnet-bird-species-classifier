import gradio as gr
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import datasets  

def load_my_model(weights_path, num_classes): 
    """Loads your trained model from the .pth weights file."""
    model = models.efficientnet_b0(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes) 
    print(f"Loading model from: {weights_path}")
    try:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_image(image, model, class_names): 
    """Performs prediction on the uploaded image using the loaded model."""
    if model is None:
        return "Model not loaded yet."
    if image is None:
        return "No image uploaded."
    try:
        img = Image.fromarray(image).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_class_index = torch.argmax(probabilities).item()

        predicted_class_name = class_names[predicted_class_index]
        return f"Predicted class: {predicted_class_name} (Probability: {probabilities[predicted_class_index].item():.4f})"
    except Exception as e:
        return f"Error during prediction: {e}"

def build_ui(model_weights_path, train_dir):
    """Builds the Gradio user interface."""
    train_dataset = datasets.ImageFolder(train_dir) 
    class_names = train_dataset.classes
    num_classes = len(class_names) 
    loaded_model = load_my_model(model_weights_path, num_classes)

    def predict_button_click(image):
        return predict_image(image, loaded_model, class_names) 

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="green")) as ui:
        gr.Markdown("# Nature Image Classifier")
        with gr.Row():
            image_input = gr.Image(label="Upload Image")
        with gr.Row():
            clear_button = gr.ClearButton([image_input], value="Clear Image")
            predict_button = gr.Button("Submit for Prediction", variant="primary")
        prediction_output = gr.Textbox(label="Prediction")

        predict_button.click(predict_button_click, inputs=[image_input], outputs=[prediction_output])

    return ui

if __name__ == "__main__":
    model_weights_file = r"C:\Users\UJJWAL\OneDrive\Desktop\BirdClassification\efficientnetb0_best.pth"
    train_dir = r"C:\Users\UJJWAL\OneDrive\Desktop\BirdClassification\Train"
    demo = build_ui(model_weights_file, train_dir)
    demo.launch()