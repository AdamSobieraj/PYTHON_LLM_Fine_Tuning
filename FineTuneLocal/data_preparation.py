import pandas as pd  # Library for data manipulation and analysis
import nltk  # Natural Language Toolkit for text processing
from nltk.tokenize import word_tokenize  # Function to split text into words
from nltk.corpus import stopwords  # Common words to filter out
from wordcloud import WordCloud  # Library to create word clouds
from collections import Counter  # Class to count occurrences of elements
import matplotlib.pyplot as plt  # Library for creating plots and visualizations
import tiktoken  # Library to calculate token counts for OpenAI models
import csv  # Library to read/write CSV files
import os  # Library for operating system interfaces
import random  # Library for random number generation
import json  # Library for handling JSON data

# Download required NLTK data (needed for text processing)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load the BBC news dataset from CSV file into a pandas DataFrame
df = pd.read_csv("data/bbc.csv")

# Display basic information about the dataset structure
print("Dataset Info:")
print(df.info())

# Show statistical summary of numerical columns
print("\nDataset Description:")
print(df.describe())

# Count how many articles belong to each category
print("\nCategory counts:")
label_counts = df['category'].value_counts()
print(label_counts)

# Show total number of text entries and unique text entries
print(f"\nTotal text entries: {len(df['text'])}")
print(f"Unique text entries: {len(df['text'].unique())}")

# Remove duplicate rows from the dataset to avoid data contamination
df.drop_duplicates(inplace=True)
print(f"After removing duplicates: {len(df['text'])} entries")

# Check for missing values in each column
null_counts = df.isnull().sum()
print("\nNull value counts:")
print(null_counts)

# Remove any remaining rows with null/missing values
df.dropna(inplace=True)

# Select only articles categorized as "tech" for word cloud generation
target_label = 'tech'
target_df = df[df['category'] == target_label]

# Process text to extract meaningful words for the tech category
# This involves tokenization, lowercasing, filtering out punctuation and stop words
target_words = [word.lower() for text in target_df['text']
                for word in word_tokenize(text)
                if word.isalnum() and word.lower() not in stopwords.words('english')]

# Count frequency of each extracted word
target_word_counts = Counter(target_words)
print(f"\nWord counts for {target_label} category:")
print(target_word_counts)

# Generate a visual word cloud based on word frequencies
word_cloud = WordCloud().generate_from_frequencies(target_word_counts)
# Display the word cloud with appropriate formatting
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.title(f"World cloud for category: {target_label}")
plt.show()

# Function to calculate token count for a single string using OpenAI's tokenizer
def cal_num_tokens_from_row(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    token_count = len(encoding.encode(string))
    return token_count

# Function to calculate total token count across all texts in a DataFrame
def cal_num_tokens_from_df(df, encoding_name: str) -> int:
    total_tokens = 0
    for text in df['text']:
        total_tokens += cal_num_tokens_from_row(text, encoding_name)
    return total_tokens

# Calculate and display total token count for the entire dataset
total_tokens = cal_num_tokens_from_df(df, "gpt-3.5-turbo")
print(f"\nTotal number of tokens in the dataframe: {total_tokens}")

# Prepare dataset for training by creating a directory to store files
os.makedirs('dataset', exist_ok=True)

# Create a list of dictionaries with clean text and corresponding labels
rows = [{'text': row['text'].strip(),
         'label': row['category'],
         } for idx, row in df.iterrows()]

# Set random seed for reproducible shuffling
random.seed(42)
# Shuffle all data rows randomly to ensure proper train/test split
random.shuffle(rows)

# Define test set size and create train/test splits
num_test = 500
splits = {'test': rows[0:num_test], 'train': rows[num_test:]}

# Write train and test datasets to separate CSV files
for split in ['train', 'test']:
    with open(f'dataset/{split}.csv', 'w', newline='', encoding='utf-8') as fOut:
        writer = csv.DictWriter(fOut, fieldnames=['text', 'label'])
        writer.writeheader()
        for row in splits[split]:
            writer.writerow(row)

# Prepare test data for evaluation by loading the test dataset
df_test = pd.read_csv("dataset/test.csv", encoding='unicode_escape')
# Take only first 500 entries for evaluation purposes
final_df = df_test.head(500)
# Calculate token count for the evaluation dataset
total_tokens_eval = cal_num_tokens_from_df(final_df, "gpt-3.5-turbo")
print(f"\nTotal number of tokens in the evaluation dataframe: {total_tokens_eval}")

# Display information about the evaluation dataset
print("\nEvaluation dataset info:")
final_df.info()

# Create test.jsonl file in OpenAI format for fine-tuning
system = "You are a intelligent assistant designed to classify news articles into these categories: business, entertainment, politics, sport, tech"
with open("test.jsonl", "w") as outfile:
    # Process each row in the evaluation dataset
    for _, row in final_df.iterrows():
        # Format data in OpenAI's required JSONL format with system prompt
        openai_format = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": row["text"]},
                {"role": "assistant", "content": row["label"]}
            ]
        }
        # Write the formatted data to file
        json.dump(openai_format, outfile)
        outfile.write('\n')  # Add newline after each JSON object