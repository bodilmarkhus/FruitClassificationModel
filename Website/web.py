import gradio as gr
import tensorflow as tf
import numpy as np

# Importing the trained model 
# NB!! the path has to be changed to reproduce the results
model = tf.keras.models.load_model('/Fruitoo/fruitoo_v3_10ep.keras')

# Names of all the fruit classifications
class_names = [
    'acai', 
    'apple', 
    'apricot', 
    'avocado', 
    'banana', 
    'blackberry', 
    'blueberry', 
    'cantaloupe', 
    'cherry', 
    'cloudberry', 
    'cocoabean', 
    'coconut', 
    'dragonfruit', 
    'fig', 
    'grape', 
    'guava', 
    'kiwi', 
    'mango', 
    'orange', 
    'papaya', 
    'passionfruit', 
    'pear', 
    'persimmon', 
    'pineapple', 
    'plumcot', 
    'pomegranate', 
    'raspberry', 
    'starfruit', 
    'strawberry', 
    'tomato'
]

def run_fruitoo(fruit):
    
    # Preprocessing the image
    fruit_array = tf.keras.preprocessing.image.smart_resize(
        fruit,
        (180, 180),
    )

    # The input image as an array for processing through the model
    fruit_array = tf.expand_dims(fruit_array, 0)
    
    # The actual predicting on the input image
    predictions = model.predict(fruit_array).flatten()
    score = tf.nn.softmax(predictions)
    
    confidences = {}

    for i in range(len(class_names)):
        confidences[class_names[i]] = float(score[i])

    # Return the predition results
    return confidences
    # (
    #    "I, the great Fruitoo, predict with a {:.2f} percent confidence that this is an image of a {}!"
    #    .format(100 * np.max(score), class_names[np.argmax(score)])
    #)

# Website interface
website = gr.Interface (
    fn = run_fruitoo, 
    inputs = gr.Image(),
    outputs = gr.Label(num_top_classes = 7),
)

# Start the website for the image prediction
website.launch()
