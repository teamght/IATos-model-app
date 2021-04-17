import tensorflow as tf
from pydub import AudioSegment
import base64
import glob
from scipy.io import wavfile
import numpy as np
import os
import tqdm
import librosa

from .config import TARGET_SAMPLE_RATE


def generate_webm_file(b64_str, directory, webmfile):
    decodedData = base64.b64decode(b64_str)
    webmfile_ = os.path.join(directory, webmfile) # También puede tener extensión .wav
    print(webmfile_)
    with open(webmfile_, 'wb') as file:
        file.write(decodedData)

def webm_to_wav(directory, webmfile, audio_name_wav):
    # Workaround para Windows 10 - Inicio
    #path = os.path.dirname(os.path.realpath(__file__))
    #os.environ["PATH"] += os.pathsep + os.path.join(path, "bin")
    #AudioSegment.converter = os.path.join(path, "bin", "ffmpeg.exe")
    #AudioSegment.ffmpeg = os.path.join(path, "bin", "ffmpeg.exe")
    #AudioSegment.ffprobe = os.path.join(path, "bin", "ffprobe.exe")
    # Workaround para Windows 10 - Fin
    
    ruta_archivo = os.path.join(directory,webmfile)
    print('Nombre de archivo WEBM: {}'.format(ruta_archivo))
    sound = AudioSegment.from_file('{}'.format(ruta_archivo) , 'webm')
    sound = sound.set_frame_rate(16000)
    sound = sound.set_sample_width(2)
    sound.export('{}'.format(os.path.join(directory, audio_name_wav)), format='wav')
    print('Nombre de archivo WAR: {0}{1}'.format(directory, audio_name_wav))

def extract_snippets(snippets_dir_x, wav_path, snippet_duration_sec=1):
    print('extract_snippets: {}'.format(wav_path))
    print('extract_snippets: {}'.format(os.path.splitext(wav_path)[0]))
    basename = os.path.basename(os.path.splitext(wav_path)[0])
    sample_rate, xs = wavfile.read(wav_path)
    print(xs.dtype)
    # Audio grabado con 256kb
    assert xs.dtype == np.int16
    # Audio grabado desde la web
    #assert xs.dtype == np.int32
    n_samples_per_snippet = int(snippet_duration_sec * sample_rate)
    i = 0
    while i + n_samples_per_snippet < len(xs):
        snippet_wav_path = os.path.join(snippets_dir_x, "%s_%.5d.wav" % (basename, i))
        snippet = xs[i : i + n_samples_per_snippet].astype(np.int16)
        #snippet = xs[i : i + n_samples_per_snippet].astype(np.int32)
        wavfile.write(snippet_wav_path, sample_rate, snippet)
        i += n_samples_per_snippet

def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)

    return parts[-2] 

def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label

AUTOTUNE = tf.data.AUTOTUNE

WORDS = np.array(['noise', 'negativo', 'positivo'])
commands = WORDS
########
def get_spectrogram(waveform):
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
    #zero_padding = tf.zeros([32000] - tf.shape(waveform), dtype=tf.float32)

    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
        
    spectrogram = tf.abs(spectrogram)

    return spectrogram

def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id

def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(
        get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
    return output_ds

def resample_wavs(data_test, target_sample_rate=16000):
    wav_paths = glob.glob(os.path.join(data_test, "*.wav"))
    print(data_test)
    resampled_suffix = "_%shz.wav" % target_sample_rate
    for i, wav_path in tqdm.tqdm(enumerate(wav_paths)):
        if wav_path.endswith(resampled_suffix):
            continue
        sample_rate, xs = wavfile.read(wav_path)
        xs = xs.astype(np.float32)
        xs = librosa.resample(xs, sample_rate, TARGET_SAMPLE_RATE).astype(np.int16)
        #xs = librosa.resample(xs, sample_rate, TARGET_SAMPLE_RATE).astype(np.int32)
        print(wav_path)
        head, tail = os.path.split(wav_path)
        print(os.path.splitext(wav_path)[0])
        print(os.path.dirname(wav_path))
        #resampled_path = '16Hz/' + os.path.splitext(wav_path)[0] + resampled_suffix
        resampled_path = head + '/16Hz/' + os.path.splitext(tail)[0] + resampled_suffix
        print(resampled_path)
        wavfile.write(resampled_path , target_sample_rate, xs)
