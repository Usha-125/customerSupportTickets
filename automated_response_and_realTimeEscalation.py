import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('stopwords')

df = pd.read_csv(r'c:/Users/hriti/infosys_internship/dataset-tickets-multi-lang3-4k.csv')

df.fillna('', inplace=True) 


df['subject'] = df['subject'].str.lower()
df['body'] = df['body'].str.lower()

stop_words = set(stopwords.words('english'))
df['subject'] = df['subject'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))
df['body'] = df['body'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))

stemmer = PorterStemmer()
df['subject'] = df['subject'].apply(lambda x: ' '.join(stemmer.stem(word) for word in x.split()))
df['body'] = df['body'].apply(lambda x: ' '.join(stemmer.stem(word) for word in x.split()))

df['text_combined'] = df['subject'] + ' ' + df['body']

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['text_combined'])


issue_counts = df['queue'].value_counts()

print("Issue Counts:")
print(issue_counts)

total_tickets = len(df)
threshold = total_tickets * 0.03

frequent_issues = issue_counts[issue_counts >= threshold]

# Display the frequent issues
print("\nFrequent Issues (occurring at least 3% of total tickets):")
print(frequent_issues)

# Set the style
sns.set(style="whitegrid")

# Plotting the issue counts
plt.figure(figsize=(12, 6))
sns.barplot(x=issue_counts.index, y=issue_counts.values, palette='viridis')
plt.title('Counts of Issue Classes')
plt.xlabel('Issue Class (Queue)')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Display details for frequent issues
for issue in frequent_issues.index:
    print(f"\nDetails for {issue}:")
    issue_details = df[df['queue'] == issue]
    print(issue_details[['subject', 'body', 'priority', 'language']].head(10))  # Adjust columns as needed

def generate_response(customer_name, product, steps, support_team):
    # Define the template
    template = """
    Dear {customer_name},

    Thank you for reaching out regarding your issue with {product}. We understand how important this matter is and appreciate your patience as we work towards a resolution.

    To assist you better, please try the following steps:
    1. {step_1}
    2. {step_2}
    3. {step_3}

    If the issue persists, please provide us with any additional details or error messages you may have encountered. We are committed to resolving this matter quickly.

    Best regards,
    {support_team}
    """
    
    # Fill in the template with actual values
    response = template.format(
        customer_name=customer_name,
        product=product,
        step_1=steps[0],
        step_2=steps[1],
        step_3=steps[2],
        support_team=support_team
    )
    
    return response




# Identify the most common issue class (queue)
issue_counts = df['queue'].value_counts()
most_common_issue = issue_counts.idxmax()  # Get the most common issue class

# Define a template response based on the most common issue
def generate_response(issue_class):
    if issue_class == "Technical Support":
        response = f"""
        Dear Customer,

        Thank you for reaching out regarding your {issue_class} issue. We understand how critical it is to resolve this matter promptly.

        To assist you better, please try the following troubleshooting steps:

        1. Ensure that your device is running the latest software updates.
        2. Check your network connection to ensure it is stable and functioning properly.
        3. If applicable, restart your device to refresh the system.

        If the problem persists after trying these steps, please provide us with any additional details or error messages you may have encountered. This information will help us diagnose the issue more effectively.

        We appreciate your patience and cooperation as we work towards a resolution.

        Best regards,
        Technical Support Team
        """
    else:
        response = f"""
        Dear Customer,

        Thank you for reaching out regarding your {issue_class} issue. We are here to assist you.

        Best regards,
        Customer Support Team
        """
    
    return response

response_email = generate_response(most_common_issue)
print(response_email)