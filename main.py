import streamlit as st
import mysql.connector
import numpy as np
import librosa
import soundfile as sf
import sounddevice as sd
from io import BytesIO
from scipy.io.wavfile import write
import torch
from silero import silero_stt, silero_tts, silero_te
from werkzeug.security import generate_password_hash, check_password_hash
import os
import time
import subprocess
import signal
import sys
# Attempt to load Intel optimizations if available
try:
    import intel_extension_for_pytorch as ipex
    ipex_installed = True
except ImportError:
    ipex_installed = False

# Detect hardware capabilities
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
elif ipex_installed:
    device = torch.device("cpu")
    print("Using Intel CPU with IPEX optimizations.")
else:
    device = torch.device("cpu")
    print("Using CPU without optimizations.")

print(f"Using device: {device}")

# Database connection
def create_connection():
    try:
        return mysql.connector.connect(
            host="localhost",  
            user="root",  
            password="vj280203",  
            database="ai_voice_me",  
            collation="utf8mb4_general_ci"
        )
    except mysql.connector.Error as err:
        st.error(f"Error connecting to database: {err}")
        return None

# User management functions
def register_user(username, password):
    connection = create_connection()
    if connection:
        cursor = connection.cursor()
        hashed_password = generate_password_hash(password)
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
            connection.commit()
        except mysql.connector.Error as err:
            st.error(f"Error registering user: {err}")
        finally:
            cursor.close()
            connection.close()

def authenticate_user(username, password):
    connection = create_connection()
    if connection:
        cursor = connection.cursor()
        cursor.execute("SELECT id, username, password FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        cursor.close()
        connection.close()
        if user and check_password_hash(user[2], password):
            return user
    return None

import io
import numpy as np
import soundfile as sf
import mysql.connector

# Store audio details in the database
def store_audio(user_id, file_name, audio_data):
    try:
        # Check if the audio_data is a numpy.ndarray (i.e., recorded audio)
        if isinstance(audio_data, np.ndarray):
            # Convert numpy array audio_data to bytes using BytesIO
            byte_io = io.BytesIO()
            # Assuming the audio is stored as a numpy array and sample rate is 96000 (high definition audio)
            sf.write(byte_io, audio_data, 96000, format='WAV')  # Adjust sample rate if necessary
            byte_io.seek(0)  # Move the pointer to the beginning of the byte stream
            audio_data_bytes = byte_io.read()
        else:
            # If audio_data is already in bytes (e.g., uploaded .wav file)
            audio_data_bytes = audio_data  # No conversion needed

        # Now audio_data_bytes contains the data to be stored in the DB
        connection = create_connection()
        if connection:
            cursor = connection.cursor()
            try:
                cursor.execute("INSERT INTO audio_history (user_id, file_name, audio_data) VALUES (%s, %s, %s)", 
                               (user_id, file_name, audio_data_bytes))
                connection.commit()
            except mysql.connector.Error as err:
                st.error(f"Error storing audio: {err}")
            finally:
                cursor.close()
                connection.close()

    except Exception as e:
        st.error(f"Error processing audio data: {e}")

def fetch_audio_history(user_id):
    connection = create_connection()
    if connection:
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT file_name FROM audio_history WHERE user_id = %s", (user_id,))
            return cursor.fetchall()
        except mysql.connector.Error as err:
            st.error(f"Error fetching audio history: {err}")
            return []
        finally:
            cursor.close()
            connection.close()

def delete_audio(user_id, file_name):
    connection = create_connection()
    if connection:
        cursor = connection.cursor()
        try:
            cursor.execute("DELETE FROM audio_history WHERE user_id = %s AND file_name = %s", (user_id, file_name))
            connection.commit()
        except mysql.connector.Error as err:
            st.error(f"Error deleting audio: {err}")
        finally:
            cursor.close()
            connection.close()

# Load voice style models
def load_voice_styles():
    try:
        models = {
            "None": None,
            "Silero STT": silero_stt(device=device)[0],
            "Silero TTS (Style A)": silero_tts(device=device, language="ru")[0],
            "Silero TTS (Style B)": silero_tts(device=device, language="ru")[0],
            "Silero TE": silero_te()[0],
        }
        return models
    except Exception as e:
        st.error(f"Error loading voice styles: {e}")
        return {"None": None}

# Audio processing functions
def adjust_pitch_and_vibrato(audio, sr, pitch_shift=0, vibrato_depth=0.5, vibrato_rate=5):
    try:
        audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=pitch_shift)
        t = np.arange(len(audio)) / sr
        vibrato = np.sin(2 * np.pi * vibrato_rate * t) * vibrato_depth
        return audio * (1 + vibrato)
    except Exception as e:
        st.error(f"Error adjusting audio: {e}")
        return audio

def save_audio_file(file_name, audio_data):
    timestamp = int(time.time())
    unique_file_name = f"{file_name}"
    file_path = os.path.join("uploads", unique_file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        with open(file_path, "wb") as f:
            f.write(audio_data)
        return file_path
    except Exception as e:
        st.error(f"Error saving audio file: {e}")
        return None

def record_audio(duration):
    try:
        st.session_state.is_recording = True
        st.write("Recording started...")
        audio = sd.rec(int(duration * 96000), samplerate=96000, channels=1)
        sd.wait()
        st.session_state.is_recording = False
        st.session_state.recorded_audio = audio.flatten()
        st.write("Recording completed!")
    except Exception as e:
        st.session_state.is_recording = False
        st.error(f"Error during recording: {e}")


def resample_audio(audio_data, sr, target_sr=96000):
    if sr is None:
        raise ValueError("Sample rate (sr) is None, cannot resample audio.")
    if sr != target_sr:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
    return audio_data

def start_flask():
    # Start Flask application in a new process
    process = subprocess.Popen(["python", "flask_api.py"])
    return process

def stop_flask(process):
    # Gracefully terminate Flask process
    process.terminate()
    process.wait()  

def main():
 flask_process = None
 try:
    print("Starting Flask application...")
    flask_process = start_flask()
    st.set_page_config(page_title="A.I Voice Me", page_icon=":microphone:", layout="wide")
    st.title("A.I. Voice Me")

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.username = None
    if 'recording_duration' not in st.session_state:
        st.session_state.recording_duration = 5  
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    if 'recorded_audio' not in st.session_state:
        st.session_state.recorded_audio = None

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.username = None

    if st.session_state.logged_in:
        st.sidebar.header("Navigation")
        st.sidebar.write(f"Logged in as: {st.session_state.username}")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.username = None
            st.rerun()

        st.write("Upload or record audio, modify it, and download the results!")
        user_id = st.session_state.user_id
        models = load_voice_styles()

        audio_option = st.radio("Choose Audio Input Method", ["Upload .wav File", "Record from Microphone", "Select from History"])
        audio_data = None
        sr = None

        if audio_option == "Upload .wav File":
            audio_file = st.file_uploader("Upload a .wav file", type=["wav"])
            if audio_file:
                try:
                    audio_data, sr = librosa.load(audio_file, sr=None)
                    st.audio(audio_file, format='audio/wav')
                    store_audio(user_id, audio_file.name, audio_file.read())
                except Exception as e:
                    st.error(f"Error processing uploaded file: {e}")

        elif audio_option == "Record from Microphone":
            st.session_state.recording_duration = st.slider("Recording Duration (seconds)", 1, 300, st.session_state.recording_duration)
            if st.button("Start Recording") and not st.session_state.is_recording:
                record_audio(st.session_state.recording_duration)

            if st.session_state.is_recording:
                st.write("Recording in progress... Click 'Stop Recording' to end early.")
              
                if st.button("Stop Recording"):
                    sd.stop()
                    st.session_state.is_recording = False
                    st.session_state.recorded_audio = None
                    st.write("Recording stopped early.")

            if st.session_state.recorded_audio is not None:
              
                sr = 96000
                st.audio(st.session_state.recorded_audio, format='audio/wav', sample_rate=96000)
                username =st.session_state.username
                file_name = f"{username}_recorded_audio_{int(time.time())}.wav"  
              
                audio_data = resample_audio(st.session_state.recorded_audio, sr=96000, target_sr=96000)
                file_path = save_audio_file(file_name, audio_data)
        
                if file_path:
                   st.write(f"Audio file saved as: {file_name}")
                   user_id = st.session_state.user_id 
                   store_audio(user_id, file_name, audio_data)  
                
                
                byte_io = io.BytesIO()
                sf.write(byte_io, audio_data, 96000, format='WAV')  
                byte_io.seek(0)  
        
              
                st.download_button("Download Recorded Audio in HD", byte_io, file_name)
   
        elif audio_option == "Select from History":
            history_files = [row[0] for row in fetch_audio_history(user_id)]
            selected_file = st.selectbox("Select from history", history_files)
           
            if selected_file:
                st.write(f"Selected file: {selected_file}")
                try:
                    connection = create_connection()
                    cursor = connection.cursor()
                    cursor.execute("SELECT audio_data FROM audio_history WHERE user_id = %s AND file_name = %s", 
                           (user_id, selected_file))
                    audio_data_bytes = cursor.fetchone()[0]
                    connection.close()

                    audio_data, sr = librosa.load(io.BytesIO(audio_data_bytes), sr=None)
                    st.audio(audio_data, format='audio/wav', sample_rate=sr)

                except Exception as e:
                     st.error(f"Error loading audio from history: {e}")

        if audio_data is not None:
            pitch_shift = st.slider("Pitch Shift (in Semitones)", -12, 12, 0)
            vibrato_depth = st.slider("Vibrato Depth", 0.0, 1.0, 0.5)
            vibrato_rate = st.slider("Vibrato Rate (Hz)", 1, 10, 5)
            voice_style = st.selectbox("Choose Voice Style", list(models.keys()))

            processed_audio = adjust_pitch_and_vibrato(audio_data, sr, pitch_shift, vibrato_depth, vibrato_rate)
            processed_audio = resample_audio(processed_audio, sr, target_sr=96000)

            output = BytesIO()
            sf.write(output, processed_audio, 96000, format='wav')
            output.seek(0)

            st.audio(output, format='audio/wav')
            st.download_button("Download Processed Audio", output, "processed_audio.wav")
        else:
            st.warning("Please Record/Select/Input The Audio First.")
    else:
        st.sidebar.header("Login or Register")
        option = st.radio("Choose Action", ["Login", "Register"])
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if option == "Register" and st.button("Register"):
            if username and password:
                register_user(username, password)
                st.success("Registration successful! Please log in.")
            else:
                st.error("Please fill in all fields.")

        elif option == "Login" and st.button("Login"):
            user = authenticate_user(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.user_id = user[0]
                st.session_state.username = user[1]
                st.success(f"Welcome, {user[1]}!")
                st.rerun()
            else:
                st.error("Invalid username or password.")
 except KeyboardInterrupt:
        print("Stopping Flask application...")
        stop_flask(flask_process)
        print("Flask application stopped.")

 finally:
        stop_flask(flask_process)


if __name__ == "__main__":
    main()
