from flask import Flask, jsonify, request, render_template, send_from_directory
from werkzeug.utils import safe_join
import os
import json
import threading
from kafka import KafkaProducer, KafkaConsumer, KafkaAdminClient

app = Flask(__name__)
file_path = "/media/maryam/Seagate Expansion Drive/dataframe/fma_large (2)"

# Initialize Kafka topics
def create_kafka_topics():
    admin_client = KafkaAdminClient(bootstrap_servers="localhost:9092")
    topics = [NewTopic(name="song-played", num_partitions=1, replication_factor=1),
              NewTopic(name="song-recommendations", num_partitions=1, replication_factor=1)]
    try:
        admin_client.create_topics(new_topics=topics, validate_only=False)
    except Exception as e:
        if 'TopicAlreadyExistsError' not in str(e):
            print(f"Error creating topics: {e}")
        else:
            print("Topics already exist, skipping creation.")

# Setup Kafka producer and consumer
producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda x: json.dumps(x).encode('utf-8'))
consumer = KafkaConsumer('song-recommendations',
                         bootstrap_servers=['localhost:9092'],
                         auto_offset_reset='earliest',
                         value_deserializer=lambda x: json.loads(x.decode('utf-8')))
consumer.subscribe(['song-recommendations'])

def kafka_consumer_thread():
    for msg in consumer:
        print("Received recommendation:", msg.value)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/songs', methods=['GET'])
def get_songs():
    songs = []
    try:
        for i in range(156):
            folder_path = os.path.join(file_path, f'{i:03d}')
            if os.path.exists(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith('.mp3'):
                        songs.append(os.path.join(f'{i:03d}', file))
                    if len(songs) >= 1000:
                        break
            if len(songs) >= 1000:
                break
    except Exception as e:
        print(f"Error fetching songs: {e}")
    return jsonify(songs)

@app.route('/audio/<path:filename>')
def audio_file(filename):
    full_path = safe_join(file_path, filename)
    if not os.path.exists(full_path):
        return "File not found", 404
    return send_from_directory(os.path.dirname(full_path), os.path.basename(full_path))

@app.route('/play_song', methods=['POST'])
def play_song():
    data = request.json
    song_path = data.get('song_path')
    if song_path:
        producer.send('song-played', {'song_path': song_path})
        # Simulate recommendation generation
        producer.send('song-recommendations', {'recommendation': song_path.replace('.mp3', '_rec.mp3')})
        producer.flush()
        print(f"Played song path sent to Kafka: {song_path}")
        return jsonify({"message": "Song is now playing", "path": song_path})
    return jsonify({"message": "No song path provided"}), 400

@app.route('/recommendations_page')
def recommendations_page():
    return render_template('recommendations.html')

@app.route('/recommendations', methods=['GET'])
def fetch_recommendations():
    messages = consumer.poll(timeout_ms=1000)
    recommendations = [msg.value['recommendation'] for tp, msgs in messages.items() for msg in msgs if msg.value and 'recommendation' in msg.value]
    return jsonify({"recommendations": recommendations})

if __name__ == '__main__':
    thread = threading.Thread(target=kafka_consumer_thread, daemon=True)
    thread.start()
    app.run(debug=True, host='0.0.0.0')

