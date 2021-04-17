import tensorflow as tf
import statistics
import librosa
from .util import *
from .config import TARGET_SAMPLE_RATE, DIRECTORY_RAW_AUDIO, DIRECTORY_WAV_AUDIO
import os
import requests
import json
from datetime import datetime
import shutil

model = tf.keras.models.load_model('saved_model')

def predict_data(data):
    b64_str = data
    
    current_date = datetime.utcnow().strftime('%Y-%m-%d_%H%M%S.%f')[:-3]
    
    directory = '{}{}'.format(DIRECTORY_RAW_AUDIO, current_date)
    
    os.makedirs(directory, exist_ok=True)
    
    webmfile = '{}{}'.format(current_date, 'audio.webm')
    audio_name_wav = '{}{}'.format(current_date, 'audio_transformado.wav')

    try:
        # Generar archivo WAV
        generate_webm_file(b64_str, directory, webmfile)
        webm_to_wav(directory, webmfile, audio_name_wav)
        
        directory_data = '{}{}'.format(DIRECTORY_WAV_AUDIO, current_date)

        os.makedirs(directory_data, exist_ok=True)

        folder_wav_paths = glob.glob(os.path.join(directory, "*.wav"))
        print(folder_wav_paths)
        snippets_dir_x = os.path.join(directory_data, '')
        os.makedirs(snippets_dir_x, exist_ok=True)

        for folder_wav_path in folder_wav_paths:
            print("Extraído de %s..." % folder_wav_path)
            extract_snippets(snippets_dir_x, folder_wav_path, snippet_duration_sec=1)

        CARPETA_ARCHIVOS_SAMPLE = os.path.join(directory_data, '16Hz')
        if not os.path.exists(CARPETA_ARCHIVOS_SAMPLE):
            os.mkdir(CARPETA_ARCHIVOS_SAMPLE)
        
        resample_wavs(directory_data, target_sample_rate=TARGET_SAMPLE_RATE)

        files = tf.io.gfile.glob(CARPETA_ARCHIVOS_SAMPLE + '/*_16000hz.wav')
        print(type(files))

        test_audio = []
        test_labels = []

        for idx, file in enumerate(files):
            print(idx, file)
            sample_file = files[idx]
            test_ds = preprocess_dataset([str(sample_file)])

            for audio, label in test_ds:
                print(type(audio))
                test_audio.append(audio.numpy())

        test_audio = np.array(test_audio)
        
        #print('data: {}'.format(type(test_audio.tolist())))
        prediction = model.predict(test_audio.tolist())
        print('prediction: {}'.format(prediction))
        # You may want to further format the prediction to make it more
        # human readable
        pred = np.argmax(prediction, axis=1)
        print('El resultado es: {}'.format(pred))

        pred_moda = statistics.mode(pred)
        print('La moda es: {}'.format(pred_moda))

        results = {0:'negativo', 1:'noise', 2:'positivo'}
        print(f'Resultado: {results[pred_moda]}')

        eliminar_archivos_temporales(directory)
        eliminar_archivos_temporales(directory_data)

        return results[pred_moda]
    except:
        eliminar_archivos_temporales(directory)
        eliminar_archivos_temporales(directory_data)
        return 'Ocurrió un error. Por favor, volver a intentar.'

def eliminar_archivos_temporales(directory):
    try:
        shutil.rmtree(directory)
    except AssertionError as error:
        print(error)
        print('Error al eliminar archivos temporales de carpeta {}'.format(directory))