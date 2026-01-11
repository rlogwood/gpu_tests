import os
from dotenv import load_dotenv
import tensorflow as tf

def load_show_env_vars():
    load_dotenv()
    print("TensorFlow environment variables:")
    print(f"TF_XLA_FLAGS={os.getenv('TF_XLA_FLAGS')}")
    print(f"TF_CPP_MIN_LOG_LEVEL={os.getenv('TF_CPP_MIN_LOG_LEVEL')}")
    print(f"TF_ENABLE_ONEDNN_OPTS={os.getenv('TF_ENABLE_ONEDNN_OPTS')}")

load_show_env_vars()
print("TF:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

