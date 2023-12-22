# import tensorflow as tf
# import numpy as np
# import json
# from PIL import Image
# import numpy as np
# import tensorflow as tf    
# from tensorflow.keras.preprocessing import image as keras_image
# from tensorflow.keras.models import load_model



# import os
# import logging
# os.chdir('S:\MY_PROJECTS\Leaf Health Diagnosis')
# import sys
# sys.path.append('S:\MY_PROJECTS\Leaf Health Diagnosis')



model_mapping = {
    "Potato": {
        "model_path": "S:\MY_PROJECTS\Potato Disease Classification\saved_models\PotatoLeaf\PotatoLeaf_model.h5",
        "class_names": ["Early Blight", "Late Blight", "Healthy"]
    },
    "Pepper": {
        "model_path": "S:\MY_PROJECTS\Potato Disease Classification\saved_models/PepperLeaf/PepperLeaf_model.h5",
        "class_names": ['Pepper bell Bacterial spot', 'Pepper bell healthy'] 
    },
    "Tomato": {
        "model_path": "S:\MY_PROJECTS\Potato Disease Classification\saved_models/TomatoLeaf/TomatoLeaf_model.h5",
        "class_names": ['Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot', 'Tomato Spider mites Two spotted spider mite', 'Tomato Target Spot', 'Tomato Tomato_YellowLeaf Curl Virus', 'Tomato Tomato mosaic virus', 'Tomato healthy']
    }
}

def select_model(plant_name):
    if plant_name not in model_mapping:
        raise ValueError("Invalid plant name. Choose among Potato, Pepper, or Tomato.")
    
    model_info = model_mapping[plant_name]
    model = load_model(model_info["model_path"])
    class_names = model_info["class_names"]

    return model, class_names

def treatment_for_disease(plant_name, predicted_class):
    with open('S:/MY_PROJECTS/Potato Disease Classification/Resources/treatments.json', 'r') as file:
        treatments = json.load(file)

    return treatments.get(plant_name, {}).get(predicted_class, "Treatment not found for this disease.")


def make_prediction(plant_name, model, class_names, img):
        # Preprocess the image
        target_size = (256, 256)  # Replace with your desired target size
        img = keras_image.array_to_img(img)
        resized_img = img.resize(target_size)
        resized_img_array = keras_image.img_to_array(resized_img)
        resized_img_array = np.expand_dims(resized_img_array, axis=0)  # Add batch dimension
        preprocessed_img = resized_img_array / 255.0  # Normalize pixel values
    
        # Make prediction
        predictions = model.predict(preprocessed_img)
    
        # Get predicted class and confidence
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = round(100 * np.max(predictions[0]), 2)
        treatment = treatment_for_disease(plant_name,predicted_class)

        return {
            'class': predicted_class,
            'confidence': float(confidence),
            'treatment': treatment
        }

loaded_models = {}


plants = ["Potato", "Pepper", "Tomato"]  
for plant in plants:
        model, class_names = select_model(plant)
        loaded_models[plant] = {"model": model, "class_names": class_names}


# Path of the image
image_path = "S:/MY_PROJECTS/Potato Disease Classification/DataSets/Pepper/test/Pepper__bell___Bacterial_spot/0d2635e7-df23-4ceb-b3ba-3af50bb58357___NREC_B.Spot 1874.JPG" # Replace with your image URL

# Read the image using PIL
image = Image.open(image_path)
plant = 'Potato'
result_dict = make_prediction(plant_name=plant, model=loaded_models[plant]["model"], class_names=loaded_models[plant]["class_names"], img=image)

print(result_dict)
