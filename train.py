from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from glob import glob
import os

IMAGE_SHAPE = (224, 224, 3)
TRAIN_DIR = "chest_xray/train"
TEST_DIR = "chest_xray/test"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "pneumonia_vgg16.h5")

os.makedirs(MODEL_DIR, exist_ok=True)

# Load pretrained VGG16 without top classification layer
base_model = VGG16(input_shape=IMAGE_SHAPE, weights="imagenet", include_top=False)

# Freeze pretrained layers
for layer in base_model.layers:
    layer.trainable = False

# Detect number of classes from training folder
classes = glob(os.path.join(TRAIN_DIR, "*"))

# Add custom classification head
x = Flatten()(base_model.output)
prediction = Dense(len(classes), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=prediction)

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

training_set = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=4,
    class_mode="categorical"
)

test_set = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    batch_size=4,
    class_mode="categorical"
)

# Train model
history = model.fit(
    training_set,
    validation_data=test_set,
    epochs=5,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)

# Save model
model.save(MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")

# Plot training history
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("outputs/training_accuracy.png")
plt.show()