import os
from dotenv import load_dotenv

load_dotenv()
import tensorflow as tf

print("TensorFlow environment variables:")
print(f"TF_XLA_FLAGS={os.getenv('TF_XLA_FLAGS')}")
print(f"TF_CPP_MIN_LOG_LEVEL={os.getenv('TF_CPP_MIN_LOG_LEVEL')}")
print(f"TF_ENABLE_ONEDNN_OPTS={os.getenv('TF_ENABLE_ONEDNN_OPTS')}")

print(f"tf.__version__: {tf.__version__}")
print(f"tf.config.list_physical_devices('GPU'):{tf.config.list_physical_devices('GPU')}")
print('GPUs:', tf.config.list_physical_devices('GPU'))
print('GPU details:', tf.config.list_physical_devices('GPU'))

# tf.debugging.set_log_device_placement(True)
#
# python -c "
# import tensorflow as tf
# print('TF version:', tf.__version__)
# print('GPUs:', tf.config.list_physical_devices('GPU'))
# print('GPU details:', tf.config.list_physical_devices('GPU'))
# "

# import tensorflow as tf
# print("GPU available:", len(tf.config.list_physical_devices('GPU')) > 0)

# # Simple matrix multiply test
# with tf.device('/GPU:0'):
#     a = tf.random.normal([1000, 1000])
#     b = tf.random.normal([1000, 1000])
#     c = tf.matmul(a, b)
#     print("Matrix multiply on GPU completed:", c.shape)


import tensorflow as tf

print('TF version:', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print('GPUs detected:', gpus)
print('GPU available:', len(gpus) > 0)

if len(gpus) > 0:
    print("✅ TESTING GPU EXECUTION...")
    try:
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
            print("✅ GPU MATRIX MULTIPLY SUCCESS:", c.shape)
    except Exception as e:
        print("❌ GPU EXECUTION FAILED:", str(e))
else:
    print("❌ NO GPUs - CPU fallback")
