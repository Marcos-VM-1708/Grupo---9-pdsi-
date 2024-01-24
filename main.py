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

from Utils import *
# -----------------------------------------------------------------------------

SIZE = (1025, 430)

# AQUI ABRIMOS DUAS MUSICAS, UMA QUE TERA SUA ESTRUTURA MANTIDA
# E OUTRA QUE SERVIRA DE ESTUDO PARA UMA APROXIMAÇÃO DA TRASFERENCIA DE ESTILO.
SOUND = "data/Music/Bbno Edamame.wav"
STYLE = "data/Music/Bach.wav"

# VISUALIZANDO O SINAL...
signal(SOUND)
# signal(STYLE)

content_img, mag_min, mag_max, phase = audio_to_imagem(SOUND, SIZE)
style_img, _, _, _ = audio_to_imagem(STYLE, SIZE)

plt.figure(figsize=(6, 5))

plt.subplot(1, 2, 1)
plt.title('signal music')
plt.imshow(content_img)

plt.subplot(1, 2, 2)
plt.title('sgnal style')
plt.imshow(style_img)

plt.show()

# -----------------------------------------------------------------------------
# CREATE MODEL...

content_np = np.array(content_img).T[None, None, :, :]
style_np = np.array(style_img).T[None, None, :, :]

content_tensor = tf.convert_to_tensor(content_np, dtype=tf.float32)
style_tensor = tf.convert_to_tensor(style_np, dtype=tf.float32)

BATCH, HEIGHT, WIDTH, CHANNELS = content_tensor.shape
FILTERS = 4096

input_shape = (HEIGHT, WIDTH, CHANNELS)

model = create_model(input_shape, FILTERS)
model.summary()

# -----------------------------------------------------------------------------
# TRAING...
content_features = model(content_tensor)
style_features = model(style_tensor)

gen_np = tf.random.normal((1, *input_shape))
gen = tf.Variable(gen_np)

steps_counter = 0

STEPS = 5000

optimizer = Adam(learning_rate=1)


for i in range(STEPS):
    with tf.GradientTape() as tape:
        tape.watch(gen)

        gen_features = model(gen)

        content_loss = get_content_loss(gen_features, content_features)
        style_loss = get_style_loss(gen_features, style_features) * 0.001

        loss = content_loss + style_loss

    gradients = tape.gradient(loss, [gen])
    optimizer.apply_gradients(zip(gradients, [gen]))

    if i % 50 == 0:
        print(f"Step: {i} | loss: {loss.numpy()} | {content_loss.numpy()} | {style_loss.numpy()}")

steps_counter += STEPS
# -----------------------------------------------------------------------------
# RESULTS...
gen_np = np.squeeze(gen.numpy()).T
gen_img = Image.fromarray(gen_np).convert('L')

plt.figure(figsize=(10, 8))

plt.subplot(1, 3, 1)
plt.title("Content")
plt.imshow(content_img)

plt.subplot(1, 3, 2)
plt.title("Style")
plt.imshow(style_img)

plt.subplot(1, 3, 3)
plt.title("Generated")
plt.imshow(gen_img)

plt.show()

x = image_to_audio(gen_img, mag_min, mag_max)

gen_img.convert('RGB').save('ouput.jpg')
np.save(f'weights.npy', gen.numpy())
sf.write(f'output.mp3', x, 22050)