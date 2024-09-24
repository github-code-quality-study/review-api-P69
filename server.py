import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def filter_reviews(self, location=None, start_date=None, end_date=None):
        filtered_reviews = []

        try:
            for review in reviews:
                if location and review["Location"] != location:
                    continue
                review_timestamp = datetime.strptime(review["Timestamp"], "%Y-%m-%d %H:%M:%S")
                start_date_parsed = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
                end_date_parsed = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

                if (start_date_parsed and review_timestamp < start_date_parsed) or \
                   (end_date_parsed and review_timestamp > end_date_parsed):
                    continue

                sentiment = self.analyze_sentiment(review["ReviewBody"])
                review["sentiment"] = sentiment
                filtered_reviews.append(review)
            filtered_reviews.sort(key=lambda x: x["sentiment"]["compound"], reverse=True)
        except Exception as e:
            print(f"Error occurred while filtering reviews: {e}")
        return filtered_reviews

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        if environ["REQUEST_METHOD"] == "GET":
            query_params = parse_qs(environ["QUERY_STRING"])
            location = query_params.get("location", [None])[0]
            start_date = query_params.get("start_date", [None])[0]
            end_date = query_params.get("end_date", [None])[0]
            filtered_reviews = self.filter_reviews(location=location, start_date=start_date, end_date=end_date)
            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")
            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            return [response_body]

        elif environ["REQUEST_METHOD"] == "POST":
            try:
                content_type = environ.get('CONTENT_TYPE', '')
                if content_type == 'application/json':
                    content_length = int(environ.get('CONTENT_LENGTH', 0))
                    post_body = environ['wsgi.input'].read(content_length)
                    new_review = json.loads(post_body.decode("utf-8"))
                else:
                    post_body = environ['wsgi.input'].read(int(environ.get('CONTENT_LENGTH', 0)))
                    post_data = parse_qs(post_body.decode("utf-8"))
                    new_review = {key: post_data[key][0] for key in post_data}

                if not new_review.get("Location") or not new_review.get("ReviewBody"):
                    raise ValueError("ReviewBody and Location are required fields.")

                # Specific location validation
                invalid_locations = ['Cupertino, California']  # Define invalid locations as needed
                if new_review.get("Location") in invalid_locations:
                    raise ValueError("Invalid location specified.")

                new_review["ReviewId"] = str(uuid.uuid4())
                new_review["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                new_review["sentiment"] = self.analyze_sentiment(new_review["ReviewBody"])
                reviews.append(new_review)
                response_body = json.dumps(new_review, indent=2).encode("utf-8")
                start_response("201 Created", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                return [response_body]
            except Exception as e:
                error_response = json.dumps({"error": str(e)}, indent=2).encode("utf-8")
                start_response("400 Bad Request", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(error_response)))
                ])
                return [error_response]
        else:
            error_response = json.dumps({"error": "Method not allowed"}, indent=2).encode("utf-8")
            start_response("405 Method Not Allowed", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(error_response)))
            ])
            return [error_response]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()
