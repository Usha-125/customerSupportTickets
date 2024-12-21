import os
from openai import OpenAI

client = OpenAI(
    api_key="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"
)

# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "system",
#             "content": """for every input sentence respond with sentiment in json like this: {
#    "thought": "<your thoughts on sentence and sentiment>",
#    "sentiment": "<neutral, positive, negative>"
# }""",
#         },
#         #     {
#         #     "role": "assistant",
#         #     "content": "Got it! Share a sentence, and I’ll respond with the sentiment analysis in the requested format.",
#         # },
#          {
#             "role": "user",
#             "content": """input: "He made me really angry, I will not talk to him ever!""",
#         },
#     ],
#     model="gpt-4o",
# )

import json

json.loads(chat_completion.choices[0].message.content)["sentiment"]

import json

def get_sentiment(title, description):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """You are a Customer Support Agent. You have to decide sentiment of the given input, using:
    1. Title
    2. Chat History

    and use 'save_sentiment' to save the same.""",
            },
            {
                "role": "user",
                "content": f"""Title: "{title}"\n\n\nChat History: "{description}" """,
            },
        ],
        functions=[
            {
                "name": "save_sentiemnt",
                "description": "Analyze the sentiment of given input.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "thought": {"type": "string", "description": "Think about the given input, and try to give it a sentiment."},
                        "sentiment": {
                            "type": "string",
                            "enum": ["positive", "negative", "neutral", "frustrated"],
                            "description": "The sentiment of the input.",
                        },
                    },
                    "required": ["thought", "sentiment"],
                },
            }
        ],
        model="gpt-4o",
    )

    if chat_completion.choices[0].finish_reason == "function_call":
        print("Thought: ", json.loads(chat_completion.choices[0].message.function_call.arguments)["thought"])
        return json.loads(chat_completion.choices[0].message.function_call.arguments)["sentiment"]
    

get_sentiment("Why is this happening!", "This does not even make sense!!!"), get_sentiment("This is an awesome product!", "Our team loves your product!!!")










import pandas as pd

# Assuming the parquet file is named 'your_file.parquet' and is in the current directory
# Replace 'your_file.parquet' with the actual file name if it is different.
try:
    df = pd.read_parquet('train-00000-of-00001-a5a7c6e4bb30b016.parquet')
    # print(df.head()) # Display the first few rows of the DataFrame
except FileNotFoundError:
    print("Error: 'your_file.parquet' not found in the current directory.")
except Exception as e:
    print(f"An error occurred: {e}")



df["customer_sentiment"].unique()

res = df.iloc[:10, :].apply(lambda x: get_sentiment(x["issue_category_sub_category"], x["conversation"]), axis=1)



# prompt: make a confusion matrix for "customer_sentiment": 'neutral', 'negative', 'frustrated', 'positive' and response (res): 'neutral', 'negative', 'frustrated', 'positive'
# a 4x4 confusion matrix

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from openai import OpenAI
from sklearn.metrics import confusion_matrix

# Assuming df is already defined and has columns 'customer_sentiment' and 'res'
# You may need to replace 'res' with the actual column containing your model's sentiment predictions

# Define categories for the confusion matrix
categories = ['frustrated', 'negative', 'neutral', 'positive']

# Ensure that your 'customer_sentiment' and 'res' columns are in the correct order and with the correct values
df_subset = df.iloc[:10, :]
# Create a mapping of string categories to numeric values
category_to_int = {cat: i for i, cat in enumerate(categories)}

# Convert string labels to numeric values
y_true = df_subset["customer_sentiment"].map(category_to_int)
y_pred = res.map(category_to_int)

# # Handle NaNs and empty lists by filtering them out
# y_true = y_true.dropna().astype(int)
# y_pred = y_pred.dropna().astype(int)

# # Filter out the corresponding indices from the dataframe where the na values are
# df_subset = df_subset.iloc[y_true.index]


# Generate the confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

# Create a Pandas DataFrame for better visualization
cm_df = pd.DataFrame(cm, index=categories, columns=categories)

# Plotting the confusion matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Sentiment')
plt.xlabel('Predicted Sentiment')
plt.show()



# prompt: flatten the list of list using pandas function: df["agent_experience_level_desc"].map(lambda x: x.split(", "))

# Assuming df["agent_experience_level_desc"] is the column you want to flatten
df["agent_experience_level_desc"] = df["agent_experience_level_desc"].str.split(", ")
# Flatten the list of lists
flattened_list = [item for sublist in df["agent_experience_level_desc"] for item in sublist]
flattened_list




# prompt: get correlation between df["customer_sentiment"]; df["issue_complexity"] both are categorical features, do label encoding

import pandas as pd

df['customer_sentiment_encoded'] = df['customer_sentiment'].map({"neutral": 0, "negative": -1, "frustrated": -2, "positive": 1})
df['issue_complexity_encoded'] = df['issue_complexity'].map({"less": 0, "medium": 1, "high": 2})

# Calculate the correlation between the encoded columns
correlation = df['customer_sentiment_encoded'].corr(df['issue_complexity_encoded'])

print(f"The correlation between customer sentiment and issue complexity is: {correlation}")


# prompt: remove stop words from string

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def remove_stopwords(text):
  stop_words = set(stopwords.words('english'))
  words = text.split()
  filtered_words = [word for word in words if word.lower() not in stop_words]
  return " ".join(filtered_words)

# prompt: get top 2 and 3 length length phrases from both input, and output phrases

import pandas as pd

def get_top_phrases(text_series, n=2):
    """
    Gets the top n most frequent phrases from a pandas Series of text.

    Args:
    text_series: A pandas Series containing text data.
    n: The number of top phrases to retrieve (default is 2).

    Returns:
    A list of tuples, where each tuple contains a phrase and its frequency.
    """
    phrase_counts = {}
    for text in text_series:
      if isinstance(text, str): # Handle potential NaN values or non-string entries
        phrases = remove_stopwords(text.lower()).split()
        for i in range(len(phrases) - 1):  # Iterate over pairs of words
          phrase = phrases[i] + " " + phrases[i+1]
          phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

        for i in range(len(phrases) - 2): # Iterate over triplets of words
          phrase = phrases[i] + " " + phrases[i + 1] + " " + phrases[i+2]
          phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
    sorted_phrases = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_phrases[:n]

# Example usage (assuming 'df' is your DataFrame and 'text_column' is your text column):
# Replace 'your_text_column' with the actual name of your column containing the text data.
# Assuming the input phrases are in the 'Ticket Description' column
input_phrases = get_top_phrases(df['input'].sample(frac=0.01), n=10)
print("Top 3 input phrases:", input_phrases)

# Assuming the output phrases are in the 'Ticket Subject' column
output_phrases = get_top_phrases(df['output'].sample(frac=0.01), n=10)
print("Top 3 output phrases:", output_phrases)


from collections import Counter

def top_10_words(text):
    words = text.lower().split()
    word_counts = Counter(words)
    return word_counts.most_common(10)

# Assuming your DataFrame is named 'df' and it contains the columns 'input' and 'output'
if 'input' in df.columns and 'output' in df.columns:
    print("Top 10 words in 'input' column:")
    for word, count in top_10_words(" ".join(df['input'].astype(str).tolist())):
        print(f"{word}: {count}")

    print("\nTop 10 words in 'output' column:")
    for word, count in top_10_words(" ".join(df['output'].astype(str).tolist())):
        print(f"{word}: {count}")
else:
    print("Error: 'input' or 'output' columns not found in the DataFrame.")





# prompt: distribution plot for length of strings in iput nad output columns

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'df' is your DataFrame and it has 'input' and 'output' columns
# Replace 'input' and 'output' with the actual column names if they are different

# Calculate the length of strings in 'input' and 'output' columns
df['input_length'] = df['input'].astype(str).apply(lambda x: len(x.split()))
df['output_length'] = df['output'].astype(str).apply(lambda x: len(x.split()))

# Create the distribution plot
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(df['input_length'], kde=True)
plt.title('Distribution of Input Length')
plt.xlabel('Length of Input Strings')
plt.ylabel('Frequency')


plt.subplot(1, 2, 2)
sns.histplot(df['output_length'], kde=True)
plt.title('Distribution of Output Length')
plt.xlabel('Length of Output Strings')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()