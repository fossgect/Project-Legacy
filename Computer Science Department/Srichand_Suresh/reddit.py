import praw # type: ignore
import re
import nltk # type: ignore
from nltk.corpus import stopwords # type: ignore
from nltk.stem import WordNetLemmatizer # type: ignore
from sklearn.feature_extraction.text import CountVectorizer # type: ignore
from sklearn.decomposition import LatentDirichletAllocation # type: ignore
import numpy as np


reddit = praw.Reddit(client_id='CLIENT_ID',
                     client_secret='CLIENT_SECRET',
                     user_agent='USER_AGENT')

stop_words = set(stopwords.words('english')).union(set(['comment', 'deleted', 'removed']))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(tokens)

def get_reddit_comments(url):
    submission = reddit.submission(url=url)
    submission.comments.replace_more(limit=0)
    comments = []
    for comment in submission.comments.list():
        comments.append(comment.body)
    return comments


def display_topics(model, feature_names, no_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topics.append(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
    return topics

def main():
    url = input("Enter the Reddit thread URL: ")
    comments = get_reddit_comments(url)
    processed_comments = [preprocess(comment) for comment in comments]
    
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', ngram_range=(1, 2))
    dtm = vectorizer.fit_transform(processed_comments)
    
    lda = LatentDirichletAllocation(n_components=5, random_state=42,learning_decay=0.7)
    lda.fit(dtm)
    
    no_top_words = 6
    topics = display_topics(lda, vectorizer.get_feature_names_out(), no_top_words)
    
    print("\nTopics in the Reddit thread:")
    for idx, topic in enumerate(topics):
        print(f"Topic {idx + 1}: {topic}")

if __name__ == '__main__':
    main()