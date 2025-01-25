
import sqlite3

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('emotions.db')
cursor = conn.cursor()

# Create table for storing emotions and content links
cursor.execute('''
CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    emotion TEXT NOT NULL,
    link TEXT NOT NULL
)
''')

# Define recommendations based on the detected emotion
recommendations = {
    'Angry': [
        "https://www.headspace.com/meditation/kids",
        "https://www.calm.com/kids",
        "Mindful Kids: 50 Mindfulness Activities",
        "https://www.youtube.com/watch?v=O29e4rRMrV4"
    ],
    'Disgust': [
        "https://www.gonoodle.com/for-families/",
        "https://kidshealth.org/en/kids/center/mindfulness/",
        "Mindful Breathing Exercises for Kids",
        "https://www.youtube.com/watch?v=cOaA7aKho3g"
    ],
    'Fear': [
        "https://www.storyberries.com/",
        "Guided Meditations for Children: Calm Kids",
        "https://www.cosmickids.com/",
        "https://www.youtube.com/watch?v=WtTGQe7VL_4"
    ],
    'Happy': [
        "https://www.pinterest.com/kidsmindfulness/",
        "Positive Songs for Kids",
        "https://www.youtube.com/watch?v=dOkyKyVFnSs",
        "https://www.smilingmind.com.au/kids"
    ],
    'Neutral': [
        "https://www.chopra.com/articles/mindfulness-for-kids",
        "Relaxing Music for Children",
        "https://www.mindful.org/mindful-parenting/",
        "https://www.youtube.com/watch?v=Cd1M9xD482s"
    ],
    'Sad': [
        "https://childmind.org/article/ways-to-help-kids-cope-with-emotions/",
        "Uplifting Stories for Kids",
        "https://www.youtube.com/watch?v=l-gQLqv9f4o",
        "https://www.kidspot.com.au/parenting/child/child-development/mindfulness-for-kids-how-to-teach-your-child-to-be-mindful/news-story/409b8994f412d7a38d4a3b6fd5366f03"
    ],
    'Surprise': [
        "https://www.gozen.com/",
        "Educational Games for Kids",
        "https://www.youtube.com/watch?v=1ZYbU82GVz4",
        "https://www.kidadl.com/articles/fun-mindfulness-activities-for-kids-to-ease-anxiety"
    ]
}

# Insert recommendations into the database
for emotion, links in recommendations.items():
    for link in links:
        cursor.execute('INSERT INTO recommendations (emotion, link) VALUES (?, ?)', (emotion, link))

# Commit the changes and close the connection
conn.commit()
conn.close()
