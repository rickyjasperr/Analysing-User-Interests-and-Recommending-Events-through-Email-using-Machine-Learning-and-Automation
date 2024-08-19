# Event Recommendation System

This is a Flask-based machine learning project that recommends events to users based on their interests and previous participation. The system uses TF-IDF and cosine similarity for matching events to user interests and employs simulated annealing to predict future interests.

## Features

- **User Profiles**: Stores user data including interests and email addresses.
- **Event Matching**: Uses TF-IDF to recommend events based on user interests.
- **Simulated Annealing**: Predicts users' future interests based on past data.
- **Email Notifications**: Sends personalized event recommendations to users.
- **Web Interface**: Simple web form to submit new events and view sent emails.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/event-recommendation-system.git
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask app:
   ```bash
   python app.py
   ```

## Usage

- Access the web interface by navigating to `http://localhost:5000/`.
- Submit new events through the form and view the results.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
