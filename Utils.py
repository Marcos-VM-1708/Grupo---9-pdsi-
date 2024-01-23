
from PIL import Image
import numpy as np
import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import tensorflow as tf
import soundfile as sf
from keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


import librosa
import numpy as np
from PIL import Image

def audio_to_imagem(arquivo_path, imagem_size):
    """
    Converte um arquivo de áudio em uma representação visual do espectro de frequência em uma imagem.

    Parâmetros:
    - caminho_arquivo (str): O caminho do arquivo de áudio a ser processado.
    - tamanho_imagem (tuple): Um par de valores [altura, largura] para especificar o tamanho da imagem resultante.

    Retorno:
    - imagem_espectro (PIL.Image.Image): Imagem em escala de cinza representando o espectro de frequência.
    - valor_min_mag (float): Valor mínimo da magnitude do espectro original.
    - valor_max_mag (float): Valor máximo da magnitude do espectro original.
    - fase_transformada (numpy.ndarray): Fase da transformada de Fourier de curto prazo.
    """

    sinal_audio, taxa_amostragem = librosa.load(arquivo_path)

    # TRANSFORMADA DE FOURIER - (STFT)...
    espectrograma = librosa.stft(sinal_audio)

    magnitude, fase_transformada = librosa.magphase(espectrograma)
    magnitude_log = np.log1p(magnitude)

    # NORMALIZAÇÃO DA MAGNITUDE...
    valor_min_mag, valor_max_mag = magnitude_log.min(), magnitude_log.max()
    magnitude_normalizada = (magnitude_log - valor_min_mag) / (valor_max_mag - valor_min_mag)

    # RESHAPE DA MAGNITUDE...
    magnitude_normalizada = magnitude_normalizada[:imagem_size[0], :imagem_size[1]]

    #  CONVERTE PARA IMAGEM...
    dados_imagem = (magnitude_normalizada * 255).astype(np.uint8)
    imagem_espectro = Image.fromarray(dados_imagem, mode='L')

    return imagem_espectro, valor_min_mag, valor_max_mag, fase_transformada


# ----------------------------------------------------------------------------------------------------------------------

def image_to_audio(img, mag_min, mag_max):
    """
    Converte uma imagem em escala de cinza para um sinal de áudio.

    Parâmetros:
    - img (PIL.Image.Image): A imagem em escala de cinza representando o espectro de frequência.
    - mag_min (float): Valor mínimo da magnitude do espectro original.
    - mag_max (float): Valor máximo da magnitude do espectro original.

    Retorno:
    - sinal_audio (numpy.ndarray): O sinal de áudio reconstruído a partir da imagem.
    """

    mag_norm = np.array(img, dtype=np.float32) / 255
    mag = mag_norm * (mag_max - mag_min) + mag_min
    mag = np.exp(mag) - 1

    # RECONSTRUIR O SINAL DE ÁUDIO USANDO O ALGORITMO GRIFFIN-LIM...
    sinal_audio = librosa.griffinlim(mag)
    return sinal_audio
# ----------------------------------------------------------------------------------------------------------------------

def custom_kernel_initializer(shape, dtype=None):
    """
    Inicializa um kernel de maneira personalizada para uma camada Conv2D.

    Parameters:
        shape (tuple): Formato do kernel.
        dtype (tf.dtypes.DType, opcional): Tipo de dados (padrão é None).

    Returns:
        tf.Tensor: Tensor constante representando o kernel inicializado.
    """
    # CALCULA DESVIO PADRÃO IDEAL...
    std = np.sqrt(2) * np.sqrt(2.0 / ((1025 + 4096) * 11)) # CHANNELS + FILTERS
    # GERA UMA MATRIZ DE DIMENSÕES (1, 11, CHANNELS, FILTERS)...
    kernel = np.random.randn(1, 11, shape[-2], shape[-1]) * std
    return tf.constant(kernel, dtype=dtype)

# ----------------------------------------------------------------------------------------------------------------------

def create_model(input_shape, FILTERS):
    """
       Cria um modelo de CNN com uma camada Conv2D.

       Parameters:
           input_shape (tuple): Formato da entrada.
           FILTERS (int): Número de filtros na camada convolucional.

       Returns:
           tf.keras.models.Model: Modelo de CNN construído.
    """

    inputs = Input(shape=input_shape)

    outputs = Conv2D(
        filters=FILTERS,
        kernel_size=(1, 11),
        padding='same',
        activation='relu',
        kernel_initializer=custom_kernel_initializer
    )(inputs)

    return Model(inputs=inputs, outputs=outputs)

# ----------------------------------------------------------------------------------------------------------------------def custom_kernel_initializer(shape, dtype=None, FILTERS, CHANNELS):

def gram_matrix(x):
    feats = tf.reshape(x, (-1, x.shape[-1]))
    return tf.matmul(tf.transpose(feats), feats)

def get_style_loss(A, B):
    gram_A = gram_matrix(A)
    gram_B = gram_matrix(B)
    return tf.sqrt(tf.reduce_sum(tf.square(gram_A - gram_B)))

def get_content_loss(A, B):
    return tf.sqrt(tf.reduce_sum(tf.square(A - B)))