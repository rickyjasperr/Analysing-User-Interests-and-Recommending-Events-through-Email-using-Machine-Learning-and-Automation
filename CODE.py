import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
import math
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load datasets
user_profiles_df = pd.read_csv('user_profiles.csv')
past_participation_df = pd.read_csv('past_participation.csv', converters={'event_ids': eval})
events_df = pd.read_csv('events.csv')
user_interactions_df = pd.read_csv('user_interactions.csv')
previous_interests_df = pd.read_csv('previous_interests.csv')

# Handle missing values in the events DataFrame
events_df['name'] = events_df['name'].fillna('Unknown Event')

# Create TF-IDF Vectorizer for event names and user interests
tfidf = TfidfVectorizer(stop_words='english')

# Fit TF-IDF model on event names
event_tfidf_matrix = tfidf.fit_transform(events_df['name'])


# Simulated Annealing to predict future interests
def simulated_annealing(user_id, previous_interests):
    interests = list(set(previous_interests_df['interest']))
    current_interest = user_profiles_df.loc[user_profiles_df['user_id'] == user_id, 'interests'].values[0]
    current_state = current_interest
    best_state = current_state
    best_score = evaluate_interest(current_state, previous_interests)
    T = 1.0
    T_min = 0.0001
    alpha = 0.9

    while T > T_min:
        i = 1
        while i <= 100:
            new_state = random.choice(interests)
            new_score = evaluate_interest(new_state, previous_interests)
            if acceptance_probability(best_score, new_score, T) > random.random():
                current_state = new_state
                best_score = new_score
            i += 1
        T = T * alpha

    return best_state


def evaluate_interest(interest, previous_interests):
    return previous_interests.count(interest)


def acceptance_probability(old_score, new_score, temperature):
    if new_score > old_score:
        return 1.0
    else:
        return math.exp((new_score - old_score) / temperature)


def get_recommendations(user_id, user_interests):
    user_tfidf_vector = tfidf.transform([user_interests])

    # Calculate cosine similarities between user interests and event names
    cosine_similarities = cosine_similarity(user_tfidf_vector, event_tfidf_matrix).flatten()

    # Sort events by cosine similarity in descending order
    sorted_event_indices = np.argsort(cosine_similarities)[::-1]

    # Filter out events that the user has already participated in or interacted with
    user_past_events = past_participation_df.loc[past_participation_df['user_id'] == user_id, 'event_ids'].values[0]
    recommended_event = None

    for event_index in sorted_event_indices:
        event_id = events_df.loc[event_index, 'event_id']
        event_name = events_df.loc[event_index, 'name']
        if event_id not in user_past_events:
            recommended_event = events_df.iloc[event_index]
            break

    return recommended_event


# Email sending function
def send_email(to_email, event):
    sender_email = "sender's_email@gmail.com"  # Replace with your email
    app_password = "app_specific_password"  # Replace with your app-specific password

    subject = "Recommended Event Based on Your Interests"
    body = (
        f"Dear User,\n\n"
        f"We have found an event that matches your interests:\n\n"
        f"Event Name: {event['name']}\n"
        f"Description: {event['description']}\n"
        f"Date: {event['date']}\n\n"
        f"Best regards,\nEvent Recommendation Team"
    )

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, app_password)
            server.sendmail(sender_email, to_email, msg.as_string())
        print(f"Email sent to {to_email}")
    except Exception as e:
        print(f"Failed to send email to {to_email}: {str(e)}")


# HTML form template
html_form = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>New Event Submission</title>
</head>
<body>
    <h1>Submit a New Event</h1>
    <form action="/submit_event" method="post">
        <label for="event_name">Event Name:</label><br>
        <input type="text" id="event_name" name="event_name"><br>
        <label for="event_date">Event Date:</label><br>
        <input type="date" id="event_date" name="event_date"><br><br>
        <input type="submit" value="Submit">
    </form>
    <h2>Entered Events</h2>
    {% if events %}
        <table border="1">
            <tr>
                <th>Event ID</th>
                <th>Name</th>
                <th>Date</th>
            </tr>
            {% for event in events %}
                <tr>
                    <td>{{ event['event_id'] }}</td>
                    <td>{{ event['name'] }}</td>
                    <td>{{ event['date'] }}</td>
                </tr>
            {% endfor %}
        </table>
    {% else %}
        <p>No events have been entered yet.</p>
    {% endif %}
</body>
</html>
"""

# HTML template for displaying results
html_results = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Event Submission Results</title>
</head>
<body>
    <h1>Event Submitted Successfully</h1>
    <h2>Emails Sent To:</h2>
    {% if user_portfolios %}
        <table border="1">
            <tr>
                <th>User ID</th>
                <th>Name</th>
                <th>Email</th>
                <th>Event Email Sent To</th>
                <th>Current Interests</th>
                <th>Past Events</th>
                <th>Predicted Interest</th>
            </tr>
            {% for portfolio in user_portfolios %}
                <tr>
                    <td>{{ portfolio['user_id'] }}</td>
                    <td>{{ portfolio['name'] }}</td>
                    <td>{{ portfolio['email'] }}</td>
                    <td>{{ portfolio['event_email_sent_to'] }}</td>
                    <td>{{ portfolio['interests'] }}</td>
                    <td>{{ portfolio['past_events'] }}</td>
                    <td>{{ portfolio['predicted_interest'] }}</td>
                </tr>
            {% endfor %}
        </table>
    {% else %}
        <p>No emails were sent.</p>
    {% endif %}
    <br>
    <a href="/">Go back to submit another event</a>
</body>
</html>
"""


# Flask route to render the HTML form
@app.route('/')
def index():
    # Load events for display
    events = events_df.to_dict(orient='records')
    return render_template_string(html_form, events=events)


# Flask route to handle form submission
@app.route('/submit_event', methods=['POST'])
def submit_event():
    global events_df
    event_name = request.form['event_name']
    event_date = request.form['event_date']

    # Append new event to events_df and save to CSV
    new_event_id = events_df['event_id'].max() + 1
    new_event = {
        'event_id': new_event_id,
        'name': event_name,
        'description': event_name,  # Use the event name as description
        'date': event_date
    }
    events_df = pd.concat([events_df, pd.DataFrame([new_event])], ignore_index=True)
    events_df.to_csv('events.csv', index=False)

    user_portfolios = []
    for _, user_row in user_profiles_df.iterrows():
        user_id = user_row['user_id']
        user_email = user_row['email']
        user_interests = user_row['interests']
        past_events = past_participation_df.loc[past_participation_df['user_id'] == user_id, 'event_ids'].values[0]

        # Check if the new event matches user's interests and send email
        new_event_tfidf_vector = tfidf.transform([event_name])
        user_tfidf_vector = tfidf.transform([user_interests])
        similarity = cosine_similarity(user_tfidf_vector, new_event_tfidf_vector).flatten()[0]

        if similarity > 0.1:  # Threshold for matching interests
            send_email(user_email, new_event)
            user_portfolios.append({
                'user_id': user_id,
                'name': user_row['name'],
                'email': user_email,
                'event_email_sent_to': event_name,
                'interests': user_interests,
                'past_events': past_events,
                'predicted_interest': simulated_annealing(user_id, user_interests)
            })

    return render_template_string(html_results, user_portfolios=user_portfolios)


if __name__ == '__main__':
    app.run(debug=True)
