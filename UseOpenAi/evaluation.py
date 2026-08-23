from openai import OpenAI  # Library to interact with OpenAI API
import pandas as pd  # Library for data manipulation and analysis

# Initialize OpenAI client to communicate with OpenAI services
client = OpenAI()

# Load the test dataset from CSV file for evaluation
df = pd.read_csv('dataset/test.csv', encoding='unicode_escape')

# Extract labels (column 1) and texts (column 0) from the dataset
labels = df.iloc[:, 1].tolist()
texts = df.iloc[:, 0].tolist()

# Use only the last 50 items for evaluation (as in original notebook)
# This ensures we're testing on a specific subset of data
texts = texts[450:]
labels = labels[450:]

# Define a function to run inference on text using the fine-tuned model
def inference_for_eval(text, m):
    completion = client.chat.completions.create(
        model=m,  # The fine-tuned model identifier
        messages=[
            {"role": "system", "content": "You are a intelligent assistant designed to classify news articles into these categories: business, entertainment, politics, sport, tech."},
            {"role": "user", "content": text}  # The input text to classify
        ]
    )
    # Return the model's predicted category
    return completion.choices[0].message.content

# Run inference on all test texts using the fine-tuned model
output = [inference_for_eval(text, "ft:gpt-3.5-turbo-0125:bis::9tt19es9") for text in texts]

# Display first 10 predictions for review
print("First 10 predictions:")
print(output[:10])

# Calculate accuracy by comparing predictions with actual labels
correct_classifications = sum(classification == label for classification, label in zip(output, labels))
total_classifications = len(labels)

# Display evaluation metrics
print(f"\nTotal classifications: {total_classifications}")
print(f"Correct classifications: {correct_classifications}")

# Calculate and display the accuracy percentage
accuracy_percentage = (correct_classifications / total_classifications) * 100
print(f"Accuracy: {accuracy_percentage:.2f}%")