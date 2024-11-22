import streamlit as st
import mysql.connector
import numpy as np
import librosa
import soundfile as sf
import sounddevice as sd
import io
import torch
import os
import time
import subprocess
import signal
import transformers
import sys
import torch
import scipy.signal as signal

from io import BytesIO
from scipy.io.wavfile import write
from werkzeug.security import generate_password_hash, check_password_hash
from torch import nn
from transformers import AutoModel





try:
    import intel_extension_for_pytorch as ipex
    ipex_installed = True
except ImportError:
    ipex_installed = False

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


def load_voice_styles():
    styles = {
        "Default": {
            "pitch_shift": 0,
            "vibrato_depth": 0.0,
            "vibrato_rate": 0.0,
            "bass_boost": 1.0
        },
        "Style 1 - Soft Singing": {
            "pitch_shift": 2,
            "vibrato_depth": 0.3,
            "vibrato_rate": 6.0,
            "bass_boost": 1.2
        },
        "Style 2 - Deep Voice": {
            "pitch_shift": -3,
            "vibrato_depth": 0.2,
            "vibrato_rate": 4.0,
            "bass_boost": 1.5
        },
        "Style 3 - Robotic Tone": {
            "pitch_shift": 0,
            "vibrato_depth": 0.1,
            "vibrato_rate": 8.0,
            "bass_boost": 1.0
        },
        "Female AI - Bright Voice": {
            "pitch_shift": 4,
            "vibrato_depth": 0.5,
            "vibrato_rate": 7.0,
            "bass_boost": 0.8
        },
        "Female AI - Calm Voice": {
            "pitch_shift": 3,
            "vibrato_depth": 0.2,
            "vibrato_rate": 4.0,
            "bass_boost": 1.0
        },
        "Male AI - Deep Voice": {
            "pitch_shift": -5,
            "vibrato_depth": 0.3,
            "vibrato_rate": 5.0,
            "bass_boost": 1.6
        },
        "Male AI - Smooth Voice": {
            "pitch_shift": -2,
            "vibrato_depth": 0.1,
            "vibrato_rate": 3.0,
            "bass_boost": 1.3
        },
        "Style 4 - High-pitched Cartoonish": {
            "pitch_shift": 7,
            "vibrato_depth": 0.6,
            "vibrato_rate": 9.0,
            "bass_boost": 0.5
        },
        "Style 5 - Grunge Rock Voice": {
            "pitch_shift": -2,
            "vibrato_depth": 0.8,
            "vibrato_rate": 3.0,
            "bass_boost": 1.8
        },
        "Style 6 - Whispery Voice": {
            "pitch_shift": 0,
            "vibrato_depth": 0.9,
            "vibrato_rate": 2.0,
            "bass_boost": 0.7
        },
        "Style 7 - Hyperactive Voice": {
            "pitch_shift": 5,
            "vibrato_depth": 0.4,
            "vibrato_rate": 6.0,
            "bass_boost": 1.1
        },
        "Style 8 - Serious News Anchor": {
            "pitch_shift": -1,
            "vibrato_depth": 0.2,
            "vibrato_rate": 2.5,
            "bass_boost": 1.2
        },
        "Style 9 - Melodic Voice": {
            "pitch_shift": 2,
            "vibrato_depth": 0.4,
            "vibrato_rate": 6.5,
            "bass_boost": 1.1
        },
        "Style 10 - Hyper-realistic Female Voice": {
            "pitch_shift": 3,
            "vibrato_depth": 0.3,
            "vibrato_rate": 5.0,
            "bass_boost": 1.0
        },
        "Style 11 - Hyper-realistic Male Voice": {
            "pitch_shift": -3,
            "vibrato_depth": 0.2,
            "vibrato_rate": 4.5,
            "bass_boost": 1.4
        }
    }
    return styles




def create_connection():
    try:
        return mysql.connector.connect(
            host="localhost",  
            user="",  
            password="",  
            database="",  
            collation="utf8mb4_general_ci"
        )
    except mysql.connector.Error as err:
        st.error(f"Error connecting to database: {err}")
        return None

def register_user(username, password):
    connection = create_connection()
    if connection:
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM users WHERE username = %s", (username,))
            result = cursor.fetchone()
            if result[0] > 0:
                return {"error": "Username already exists. Please choose a different username."}
            
            hashed_password = generate_password_hash(password)
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
            connection.commit()
            return {"message": "User registered successfully"}
        except mysql.connector.Error as err:
            st.error(f"Error registering user: {err}")
            return {"error": f"Error registering user: {err}"}
        finally:
            cursor.close()
            connection.close()
    else:
        return {"error": "Failed to connect to the database."}

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


def store_audio(user_id, file_name, audio_data):
    try:
        if isinstance(audio_data, np.ndarray):
            byte_io = io.BytesIO()
            sf.write(byte_io, audio_data, 96000, format='WAV')  
            byte_io.seek(0)  
            audio_data_bytes = byte_io.read()
        else:
            audio_data_bytes = audio_data  

        connection = create_connection()
        if connection:
            cursor = connection.cursor()
            try:
                cursor.execute("SELECT COUNT(*) FROM audio_history WHERE user_id = %s AND file_name = %s", 
                               (user_id, file_name))
                result = cursor.fetchone()
                
                if result[0] > 0:
                    st.warning(f"The file '{file_name}' already exists. Consider change or renaming or updating.")
                else:
                    cursor.execute("INSERT INTO audio_history (user_id, file_name, audio_data) VALUES (%s, %s, %s)", 
                                   (user_id, file_name, audio_data_bytes))
                    connection.commit()
                    st.success("Audio stored successfully.")
            except mysql.connector.Error as err:
                st.error(f"Error storing audio: {err}")
            finally:
                cursor.close()
                connection.close()

    except Exception as e:
        st.error(f"Error processing audio data: {e}")

def store_processed_audio(user_id, filename, audio_data):
    try:
        connection = create_connection()
        cursor = connection.cursor()
        
        byte_data = audio_data.tobytes()
        
        cursor.execute(
            "INSERT INTO processed_audio (user_id, file_name, audio_data) VALUES (%s, %s, %s)",
            (user_id, filename, byte_data)
        )
        
        connection.commit()
        connection.close()
        return True
    except Exception as e:
        st.error(f"Error saving processed audio to database: {e}")
        return False

def is_processed_audio_saved(user_id, filename):
    try:
        connection = create_connection()
        cursor = connection.cursor()
        
        cursor.execute(
            "SELECT COUNT(*) FROM processed_audio WHERE user_id = %s AND file_name = %s", 
            (user_id, filename)
        )
        result = cursor.fetchone()
        connection.close()
        
        if result[0] > 0:
            return True
        return False
    except Exception as e:
        st.error(f"Error checking processed audio in database: {e}")
        return False

def is_audio_saved_locally(username, filename):
    file_path = f"uploads/{username}/{filename}"
    return os.path.exists(file_path)


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

def delete_audio_from_history(user_id, file_name, username):
    try:
        connection = create_connection()
        if connection:
            cursor = connection.cursor()
            cursor.execute("DELETE FROM audio_history WHERE user_id = %s AND file_name = %s", (user_id, file_name))
            connection.commit()

            file_path = os.path.join("uploads", username, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
                st.success(f"Audio file '{file_name}' has been deleted from history and storage.")
                st.rerun()
            else:
                st.warning(f"Audio file '{file_name}' not found in storage.")
            
            connection.close()
    except mysql.connector.Error as err:
        st.error(f"Error deleting audio file from history: {err}")
    except Exception as e:
        st.error(f"Error deleting audio file: {e}")

def adjust_pitch_and_vibrato(audio, sr, pitch_shift=0, vibrato_depth=0.5, vibrato_rate=5):
    try:
        audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=pitch_shift)
        t = np.arange(len(audio)) / sr
        vibrato = np.sin(2 * np.pi * vibrato_rate * t) * vibrato_depth
        return audio * (1 + vibrato)
    except Exception as e:
        st.error(f"Error adjusting audio: {e}")
        return audio

def apply_bass_boost(audio_data, bass_boost, sr=96000):
    """
    Apply bass boost effect to the audio data.
    """
    nyquist = 0.5 * sr  
    low = 100 / nyquist  
    b, a = signal.butter(1, low, btype='low')  
    filtered_audio = signal.filtfilt(b, a, audio_data)
    boosted_audio = audio_data + (filtered_audio * (bass_boost - 1.0))  # Apply bass boost

    return boosted_audio

import librosa

def apply_voice_conversion(audio_data, sr, voice_style, target_sr=96000):
    try:
        if sr != target_sr:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr  
        style_params = load_voice_styles().get(voice_style, {})
        pitch_shift = style_params.get("pitch_shift", 0)
        vibrato_depth = style_params.get("vibrato_depth", 0.0)
        vibrato_rate = style_params.get("vibrato_rate", 0.0)
        bass_boost = style_params.get("bass_boost", 1.0)

        audio_data = adjust_pitch_and_vibrato(audio_data, sr, pitch_shift, vibrato_depth, vibrato_rate)
        audio_data = apply_bass_boost(audio_data, bass_boost, sr)

        return audio_data
    except Exception as e:
        st.error(f"Error during voice conversion: {e}")
        return audio_data

def save_audio_file(file_name, audio_data, username):
    user_folder = username  
    timestamp = int(time.time())
    unique_file_name = f"{file_name}"
    
    file_path = os.path.join("uploads", user_folder, unique_file_name)
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

    if audio_data is None:
        raise ValueError("Audio data is None, cannot resample.")
    if sr is None:
        raise ValueError("Sample rate (sr) is None, cannot resample audio.")
    
    if sr != target_sr:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
    
    return audio_data

def start_flask():
    process = subprocess.Popen(["python", "flask_api.py"])
    return process

def stop_flask(process):
    process.terminate()
    process.wait()  

def main():
 flask_process = None
 try:
    print("Starting Flask application...")
    flask_process = start_flask()
    st.set_page_config(page_title="Voice Me", page_icon=":microphone:", layout="wide")
    st.title("Voice Me")

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
                    username = st.session_state.username
                    file_name = f"{username}_uploaded_audio_{int(time.time())}.wav"  
                    file_path = save_audio_file(file_name, audio_file.read(), username)
                    if file_path:
                      st.write(f"Audio file saved as: {file_name}")
                      user_id = st.session_state.user_id 
                      store_audio(user_id, file_name, audio_file.read())  
                
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
                username = st.session_state.username
                file_name = f"{username}_recorded_audio_{int(time.time())}.wav"  
              
                audio_data = resample_audio(st.session_state.recorded_audio, sr=96000, target_sr=96000)
                file_path = save_audio_file(file_name, audio_data, username)
        
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
                    user_audio_path = f"uploads/{st.session_state.username}/{selected_file}"
                   
                   
                    if os.path.exists(user_audio_path):
                        try:
                   
                            audio_data, sr = librosa.load(user_audio_path, sr=None)
                    
                            delete_button = st.button("Delete Audio from History and Storage")
                            if delete_button:
                               delete_audio_from_history(user_id, selected_file, st.session_state.username)
  
                            st.audio(audio_data, format='audio/wav', sample_rate=sr)

                        except Exception as e:
                         st.error(f"Error loading audio with librosa: {e}")
                    else:
                        st.error(f"Audio file {selected_file} not found in the specified folder.")
        
                except Exception as e:
                     st.error(f"Error loading audio from history: {e}")

        if audio_data is not None:
           
           voice_style = st.selectbox("Choose Voice Style", list(load_voice_styles().keys()))
           if voice_style != "None":

            style_params = load_voice_styles()[voice_style]

            pitch_shift = style_params["pitch_shift"]
            vibrato_depth = style_params["vibrato_depth"]
            vibrato_rate = style_params["vibrato_rate"]
            bass_boost = style_params["bass_boost"]


            pitch_shift = st.slider("Pitch Shift (in Semitones)", -12, 12, pitch_shift)
            vibrato_depth = st.slider("Vibrato Depth", 0.0, 1.0, vibrato_depth)
            vibrato_rate = st.slider("Vibrato Rate (Hz)", 1.0, 10.0, vibrato_rate, step=0.1)
            bass_boost = st.slider("Bass Boost", 0.5, 2.0, bass_boost, step=0.1)

            processed_audio = apply_voice_conversion(audio_data, sr, voice_style, 96000)
            
            processed_audio = adjust_pitch_and_vibrato(audio_data, sr, pitch_shift, vibrato_depth, vibrato_rate)
            
            processed_audio = apply_bass_boost(processed_audio, bass_boost)

            processed_audio = resample_audio(processed_audio, sr, target_sr=96000)

            output = BytesIO()
            sf.write(output, processed_audio, 96000, format='WAV')
            output.seek(0)
            
            filename = f"{st.session_state.username}_{voice_style.replace(' ', '_')}_processed_audio.wav"
            st.audio(output, format='audio/wav')
            st.download_button("Download Processed Audio", output, file_name=filename)
       
            if is_processed_audio_saved(st.session_state.user_id, filename) or is_audio_saved_locally(st.session_state.username, filename):
               st.warning(f"The file {filename} has already been saved.")
               save_button_disabled = True 
               st.rerun()
            else:
               save_button_disabled = False  
               st.rerun()
            
            save_button = st.button("Save Processed Audio", disabled=save_button_disabled)
           
            if save_button:
                save_audio_path = f"uploads/{st.session_state.username}/{filename}"
                os.makedirs(os.path.dirname(save_audio_path), exist_ok=True)
              
                try: 
                   sf.write(save_audio_path, processed_audio, 96000, format='WAV')
                   
                   if store_processed_audio(st.session_state.user_id, filename, processed_audio):
                    st.success(f"Processed audio saved as {filename}")
                
                except Exception as e:
                    st.error(f"Error saving processed audio: {e}")

        else:
            st.warning("Please Record/Select/Input The Audio First.")
    else:
        st.sidebar.header("Login or Register")
        option = st.radio("Choose Action", ["Login", "Register"])
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if option == "Register" and st.button("Register"):
            if username and password:
                response = register_user(username, password)
                if 'error' in response:
                    st.error(response['error'])
                else:
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
