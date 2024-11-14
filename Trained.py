from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

# Load the trained model and tokenizer
model = BertForSequenceClassification.from_pretrained("./human_vs_ai_model")
tokenizer = BertTokenizer.from_pretrained("./human_vs_ai_model")

# Function to evaluate whether the input essay is AI-generated or human-written
def detect_ai_generated_text(input_text):
    model.eval()  # Set the model to evaluation mode
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Make sure to use GPU if available
    if torch.cuda.is_available():
        inputs = {key: value.cuda() for key, value in inputs.items()}
        model.cuda()
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Apply softmax to get probabilities for each class
        probabilities = F.softmax(logits, dim=1)
        
        # Get the predicted class (highest probability)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        
        # Get the confidence (probability) of the predicted class
        confidence = probabilities[0][predicted_class].item()
    
    # Return whether the text is AI-generated (1) or Human-written (0), along with the confidence
    if predicted_class == 1:
        return f"AI-generated with {confidence*100:.2f}% confidence"
    else:
        return f"Human-written with {confidence*100:.2f}% confidence"

# Main loop for continuous input
while True:
    # Take user input
    input_text = input("Enter an essay to check if it's AI-generated (or type 'exit' to quit): ")
    
    # If the user types 'exit', break the loop
    if input_text.lower() == "exit":
        print("Exiting the program.")
        break
    
    # Get the result of the prediction
    result = detect_ai_generated_text(input_text)
    
    # Display the result
    print(f"The essay is: {result}")
