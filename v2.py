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

from demucs import pretrained
from demucs.audio import AudioFile
from demucs.apply import apply_model 
from demucs.pretrained import get_model




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



def display_saved_processed_files():
    """
    This function fetches the list of processed audio files from the database or the user's folder
    and displays them for playback or deletion.
    """
    user_id = st.session_state.user_id
    user_folder = f"uploads/{st.session_state.username}/"
    
    saved_files = fetch_saved_processed_files(user_id)  

    if saved_files:
        st.write("### Saved Processed Files:")
        
        for file in saved_files:
            st.write(file)
            
            file_path = f"{user_folder}/{file}"
            if os.path.exists(file_path):
                audio_data, sr = librosa.load(file_path, sr=None)
                st.audio(audio_data, format='audio/wav', sample_rate=sr)

            if st.button(f"Delete {file}"):
                if delete_saved_processed_audio(user_id, file):
                    st.success(f"{file} has been deleted successfully.")
                    st.experimental_rerun()
                else:
                    st.error(f"Failed to delete {file}.")
    else:
        st.write("No processed audio files found.")

def fetch_saved_processed_files(user_id):

    try:
        connection = create_connection()
        cursor = connection.cursor()
        
        cursor.execute("SELECT file_name FROM processed_audio WHERE user_id = %s", (user_id,))
        files = cursor.fetchall()
        connection.close()

        if files:
            return [file[0] for file in files]
        else:
            user_folder = f"uploads/{st.session_state.username}/"
            return [f for f in os.listdir(user_folder) if f.endswith(".wav")]
    except Exception as e:
        st.error(f"Error fetching saved files: {e}")
        return []

def delete_saved_processed_audio(user_id, filename):

    try:
        connection = create_connection()
        cursor = connection.cursor()
        
        cursor.execute("DELETE FROM processed_audio WHERE user_id = %s AND file_name = %s", (user_id, filename))
        connection.commit()

        file_path = f"uploads/{st.session_state.username}/{filename}"
        if os.path.exists(file_path):
            os.remove(file_path)

        connection.close()
        return True
    except Exception as e:
        st.error(f"Error deleting processed audio: {e}")
        return False



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
    boosted_audio = audio_data + (filtered_audio * (bass_boost - 1.0))  

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

def boost_volume(audio_data, factor=2.0):
    """Boost the audio volume by a given factor."""
    return np.clip(audio_data * factor, -1.0, 1.0)

def dynamic_gain(audio_data, target_peak=0.8):
    """Adjust audio gain dynamically to a target peak level."""
    peak = np.max(np.abs(audio_data))
    if peak == 0:
        return audio_data  

    gain = target_peak / peak
    adjusted_audio = audio_data * gain
    return np.clip(adjusted_audio, -1.0, 1.0)

def denoise_audio(audio_data, sr):

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = pretrained.get_model('htdemucs')  
        model.to(device)
        model.eval()  
        is_mono = len(audio_data.shape) == 1
        if is_mono:
            st.write("Mono Audio Detected")
        else:
            st.write("Stereo Audio Detected")

        if is_mono:
            audio_data = np.stack([audio_data, audio_data], axis=0) 
        audio_data = np.clip(audio_data, -1.0, 1.0)

        tensor_audio = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0).to(device)

        # Apply HTDemucs
        with torch.no_grad():
            sources = apply_model(model, tensor_audio, device=device)  # Output shape: (batch, stems, channels, samples)

        # Extract the vocals or primary clean signal
        enhanced_audio = sources[0, 0].cpu().numpy()  # Extract the "vocals" stem (channel 0)

        max_val = np.max(np.abs(enhanced_audio))
        if max_val > 0:
            enhanced_audio /= max_val

        if is_mono:
            return enhanced_audio[0]
        else:
            return enhanced_audio

    except Exception as e:
        st.error(f"Error during denoising: {e}")
        print(f"Error in denoise_audio: {e}")
        return audio_data

def audio_enhancement_ui():
    st.write("### Audio Enhancement")
    
    source_option = st.radio("Choose Source", ["History Audio", "Saved Processed Audio"])

    if source_option == "History Audio":
        user_id = st.session_state.user_id
        history_files = [row[0] for row in fetch_audio_history(user_id)]
        selected_file = st.selectbox("Select from History", history_files)
    elif source_option == "Saved Processed Audio":
        user_id = st.session_state.user_id
        saved_files = fetch_saved_processed_files(user_id)
        selected_file = st.selectbox("Select from Saved Processed Files", saved_files)

    if selected_file:
        audio_path = f"uploads/{st.session_state.username}/{selected_file}"
        if os.path.exists(audio_path):
            audio_data, sr = librosa.load(audio_path, sr=None)
            st.audio(audio_data, format='audio/wav', sample_rate=sr)

            st.write("#### Enhancement Options")
            denoise = st.checkbox("Apply Noise Reduction")

            if st.button("Process Audio"):
                if denoise:
                    enhanced_audio = denoise_audio(audio_data, sr)
                    st.write(f"Enhanced audio shape: {enhanced_audio.shape}, dtype: {enhanced_audio.dtype}")
                else:
                    enhanced_audio = audio_data

                try:
                    if len(enhanced_audio.shape) == 1:  
                        enhanced_audio = enhanced_audio[np.newaxis, :] 
                    elif enhanced_audio.shape[0] == 2: 
                        enhanced_audio = enhanced_audio.T  

                    enhanced_audio = np.float32(enhanced_audio)

                    output = BytesIO() 
                    st.write(f"Enhanced audio for saving, shape: {enhanced_audio.shape}, dtype: {enhanced_audio.dtype}")

                    sf.write(output, enhanced_audio.T, sr, format='WAV') 
                    output.seek(0) 
                    st.audio(output, format='audio/wav')  

                    filename = f"{selected_file.split('.')[0]}_enhanced.wav"
                    st.download_button("Download Enhanced Audio", output, file_name=filename)

                    if st.button("Save Enhanced Audio"):
                        save_audio_path = f"uploads/{st.session_state.username}/{filename}"
                        os.makedirs(os.path.dirname(save_audio_path), exist_ok=True)
                        try:
                            sf.write(save_audio_path, enhanced_audio.T, sr, format='WAV')
                            store_processed_audio(st.session_state.user_id, filename, enhanced_audio)
                            st.success(f"Enhanced audio saved as {filename}")
                        except Exception as e:
                            st.error(f"Error saving enhanced audio: {e}")
                except Exception as e:
                    st.error(f"Error during audio processing: {e}")
        else:
            st.error(f"Audio file {selected_file} not found.")



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
        if 'audio_data' not in st.session_state:
            st.session_state.audio_data = None
        if 'sr' not in st.session_state:
            st.session_state.sr = None

        if st.session_state.logged_in:
            st.sidebar.header("Navigation")
            st.sidebar.write(f"Logged in as: {st.session_state.username}")
            
            menu_options = st.sidebar.radio("Menu", ["Home", "Upload Audio", "Record Audio", "Audio Enhancement", "History", "Saved Audio"])
            
            if st.sidebar.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.user_id = None
                st.session_state.username = None
                st.rerun()

            if menu_options == "Home":
                st.write("Welcome to Voice Me!")
                st.write("Upload or record audio, modify it, and download the results!")

            elif menu_options == "Upload Audio":
                st.write("Upload a .wav file below:")
                audio_file = st.file_uploader("Upload a .wav file", type=["wav"])
                if audio_file:
                    try:
                        st.session_state.audio_data, st.session_state.sr = librosa.load(audio_file, sr=None)
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

            elif menu_options == "Record Audio":
                st.write("Record your audio using the microphone:")
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
                    st.session_state.sr = 96000
                    st.audio(st.session_state.recorded_audio, format='audio/wav', sample_rate=96000)
                    username = st.session_state.username
                    file_name = f"{username}_recorded_audio_{int(time.time())}.wav"
                    st.session_state.audio_data = resample_audio(st.session_state.recorded_audio, sr=96000, target_sr=96000)
                    file_path = save_audio_file(file_name, st.session_state.audio_data, username)
                    if file_path:
                        st.write(f"Audio file saved as: {file_name}")
                        user_id = st.session_state.user_id
                        store_audio(user_id, file_name, st.session_state.audio_data)
                    byte_io = io.BytesIO()
                    sf.write(byte_io, st.session_state.audio_data, 96000, format='WAV')
                    byte_io.seek(0)
                    st.download_button("Download Recorded Audio in HD", byte_io, file_name)

            elif menu_options == "Audio Enhancement":
                 audio_enhancement_ui()


            elif menu_options == "History":
                st.write("Select and manage your audio history:")
                user_id = st.session_state.user_id
                history_files = [row[0] for row in fetch_audio_history(user_id)]
                selected_file = st.selectbox("Select from history", history_files)
                if selected_file:
                    st.write(f"Selected file: {selected_file}")
                    try:
                        user_audio_path = f"uploads/{st.session_state.username}/{selected_file}"
                        if os.path.exists(user_audio_path):
                            try:
                                st.session_state.audio_data, st.session_state.sr = librosa.load(user_audio_path, sr=None)
                                delete_button = st.button("Delete Audio from History and Storage")
                                if delete_button:
                                    delete_audio_from_history(user_id, selected_file, st.session_state.username)
                                st.audio(st.session_state.audio_data, format='audio/wav', sample_rate=st.session_state.sr)
                            except Exception as e:
                                st.error(f"Error loading audio with librosa: {e}")
                        else:
                            st.error(f"Audio file {selected_file} not found in the specified folder.")
                    except Exception as e:
                        st.error(f"Error loading audio from history: {e}")

            elif menu_options == "Saved Audio":
                st.write("Your saved processed audio:")
                saved_audio_files = fetch_saved_processed_files(st.session_state.user_id)
                if saved_audio_files:
                    selected_saved_file = st.selectbox("Select a saved audio file", saved_audio_files)
                    if selected_saved_file:
                        st.write(f"Selected file: {selected_saved_file}")
                        saved_audio_path = f"uploads/{st.session_state.username}/{selected_saved_file}"
                        if os.path.exists(saved_audio_path):
                            try:
                                processed_audio, sr = librosa.load(saved_audio_path, sr=None)
                                st.audio(processed_audio, format='audio/wav', sample_rate=sr)
                                delete_button = st.button("Delete Saved Audio")
                                if delete_button:
                                    delete_saved_audio(st.session_state.user_id, selected_saved_file)
                                    os.remove(saved_audio_path)
                                    st.success(f"{selected_saved_file} has been deleted.")
                            except Exception as e:
                                st.error(f"Error loading saved audio file: {e}")
                else:
                    st.write("No saved audio files found.")

            if st.session_state.audio_data is not None and menu_options != "Audio Enhancement":
                st.write("Process your audio below:")
                voice_style = st.selectbox("Choose Voice Style", list(load_voice_styles().keys()))
                if voice_style != "None":
                    style_params = load_voice_styles()[voice_style]
                    pitch_shift = st.slider("Pitch Shift (in Semitones)", -12, 12, style_params["pitch_shift"])
                    vibrato_depth = st.slider("Vibrato Depth", 0.0, 1.0, style_params["vibrato_depth"])
                    vibrato_rate = st.slider("Vibrato Rate (Hz)", 1.0, 10.0, style_params["vibrato_rate"], step=0.1)
                    bass_boost = st.slider("Bass Boost", 0.5, 2.0, style_params["bass_boost"], step=0.1)

                    processed_audio = adjust_pitch_and_vibrato(
                        st.session_state.audio_data,
                        st.session_state.sr,
                        pitch_shift,
                        vibrato_depth,
                        vibrato_rate
                    )
                    processed_audio = apply_bass_boost(processed_audio, bass_boost)
                    processed_audio = resample_audio(processed_audio, st.session_state.sr, target_sr=96000)

                    output = BytesIO()
                    sf.write(output, processed_audio, 96000, format='WAV')
                    output.seek(0)
                    filename = f"{st.session_state.username}_{voice_style.replace(' ', '_')}_processed_audio.wav"
                    st.audio(output, format='audio/wav')
                    st.download_button("Download Processed Audio", output, file_name=filename)
                    if st.button("Save Processed Audio"):
                        save_audio_path = f"uploads/{st.session_state.username}/{filename}"
                        os.makedirs(os.path.dirname(save_audio_path), exist_ok=True)
                        try:
                            sf.write(save_audio_path, processed_audio, 96000, format='WAV')
                            if store_processed_audio(st.session_state.user_id, filename, processed_audio):
                                st.success(f"Processed audio saved as {filename}")
                        except Exception as e:
                            st.error(f"Error saving processed audio: {e}")

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

