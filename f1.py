import tkinter as tk
from tkinter import font, ttk
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the saved models and tokenizers
tokenizer_arabert = AutoTokenizer.from_pretrained("model_arabert")
model_arabert = AutoModelForSequenceClassification.from_pretrained("model_arabert")

tokenizer_arbert = AutoTokenizer.from_pretrained("model_arabert")
model_arbert = AutoModelForSequenceClassification.from_pretrained("model_arabert")

tokenizer_marbert = AutoTokenizer.from_pretrained("model_marbert")
model_marbert = AutoModelForSequenceClassification.from_pretrained("model_marbert")

# Function to predict sentiment
def predict(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1)
    return prediction.item()

# Voting mechanism
def majority_vote(predictions):
    counts = np.bincount(predictions)
    return np.argmax(counts)

def is_arabic(text):
    return all('\u0600' <= char <= '\u06FF' or char.isspace() for char in text)

# Function to get sentiment prediction and display results
def get_sentiment():
    text = entry.get("1.0", tk.END).strip()
    
    if not text:
        label.config(text="Please enter a sentence.", fg="red")
        return
    
    if not is_arabic(text):
        label.config(text="Please enter a sentence in Arabic.", fg="red")
        return
    
    label.config(text="Enter a sentence:", fg="black")

    prediction_arabert = predict(model_arabert, tokenizer_arabert, text)
    prediction_arbert = predict(model_arbert, tokenizer_arabert, text)
    prediction_marbert = predict(model_marbert, tokenizer_marbert, text)

    # Display individual model results
    sentiment_arabert = "Positive" if prediction_arabert == 1 else "Negative"
    sentiment_arbert = "Positive" if prediction_arbert == 1 else "Negative"
    sentiment_marbert = "Positive" if prediction_marbert == 1 else "Negative"

    result_arabert_label.config(
        text=f"Model 1 (AraBERT): {sentiment_arabert}",
        fg="green" if sentiment_arabert == "Positive" else "red"
    )
    result_arbert_label.config(
        text=f"Model 2 (ARBERT): {sentiment_arbert}",
        fg="green" if sentiment_arabert == "Positive" else "red"
    )
    result_marbert_label.config(
        text=f"Model 3 (MARBERT): {sentiment_marbert}",
        fg="green" if sentiment_marbert == "Positive" else "red"
    )

    # Calculate and display final prediction
    final_prediction = majority_vote([prediction_arabert, prediction_arbert, prediction_marbert])
    final_sentiment = "Positive" if final_prediction == 1 else "Negative"
    final_result_label.config(
        text=f"Final Prediction: {final_sentiment}",
        fg="green" if final_sentiment == "Positive" else "red"
    )

# Set up the Tkinter GUI
root = tk.Tk()
root.title("Arabic Sentiment Analysis")
root.geometry("600x400")  # Set a fixed window size

# Define colors
bg_color = "#e6f2ff"  # Light blue background
button_color = "#007bff"  # Blue button
button_text_color = "#ffffff"  # White button text
highlight_color = "#ffeb3b"  # Yellow for highlights

# Configure the root window
root.configure(bg=bg_color)

# Define fonts
title_font = font.Font(family="Helvetica", size=16, weight="bold")
label_font = font.Font(family="Helvetica", size=12)
button_font = font.Font(family="Helvetica", size=12, weight="bold")

# Create and style widgets
label = tk.Label(root, text="Enter a sentence:", bg=bg_color, fg="#333333", font=title_font)
label.pack(pady=10)

entry = tk.Text(root, height=4, width=50, wrap=tk.WORD, bg="#ffffff", fg="#333333", font=label_font, padx=10, pady=10, borderwidth=2, relief="groove")
entry.pack(pady=10)

predict_button = tk.Button(root, text="Predict Sentiment", command=get_sentiment, bg=button_color, fg=button_text_color, font=button_font, relief="raised", padx=10, pady=5)
predict_button.pack(pady=10)

result_arabert_label = tk.Label(root, text="", bg=bg_color, fg="#333333", font=label_font)
result_arabert_label.pack(pady=5)

result_arbert_label = tk.Label(root, text="", bg=bg_color, fg="#333333", font=label_font)
result_arbert_label.pack(pady=5)
    
result_marbert_label = tk.Label(root, text="", bg=bg_color, fg="#333333", font=label_font)
result_marbert_label.pack(pady=5)

final_result_label = tk.Label(root, text="", bg=bg_color, fg="#333333", font=title_font)
final_result_label.pack(pady=20)

root.mainloop()