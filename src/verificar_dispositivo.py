
import tensorflow as tf

print("Versión de TensorFlow:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("GPU disponible")
else:
    print("GPU no disponible. Se utilizará CPU.")

a = tf.ones((100, 100))
b = tf.ones((100, 100))

resultado = tf.matmul(a, b)

print("Resultado:", resultado[0, 0].numpy())
print("Dispositivo utilizado:", resultado.device)
