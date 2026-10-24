
# Face Matching & Analysis 👤

**Completed:** January 30, 2026

Face Matching & Analysis is a Python computer vision application designed to locate a particular person's face across multiple photographs.

The program uses the `face_recognition` library to detect faces, generate numerical face encodings, and compare them with a reference image. When a potential match is detected, pretrained deep learning models are used to estimate an age group and a binary male/female category.

Matching photographs are saved with a rectangle drawn around the detected face and labels showing the predictions.

I developed this project to explore face recognition, image processing, neural networks, and organizing more complex Python applications across multiple modules.

## Features

- **Face detection:** Automatically detect faces in photographs.
- **Reference-based matching:** Search a folder for faces resembling the face in a selected reference image.
- **Multiple-face processing:** Examine multiple faces within the same photograph.
- **Face encodings:** Represent faces using 128-dimensional numerical vectors.
- **Face similarity comparison:** Use face distance to determine potential matches.
- **Age-group prediction:** Classify matching faces into one of eight predefined age ranges.
- **Gender-category prediction:** Use a pretrained classifier with Male and Female output categories.
- **Automatic annotation:** Draw green rectangles and predicted labels on matching images.
- **Automatic folder creation:** Create a directory for saving matching results.
- **Progress reporting:** Display progress messages while processing photographs.
- **Modular architecture:** Separate detection, matching, prediction, and visualization functionality into different Python modules.

## Requirements

- Python 3
- Windows
- OpenCV
- `face_recognition`
- NumPy
- Pretrained Caffe models for age and gender classification

Install the Python dependencies:

```bash
python -m pip install face-recognition opencv-python numpy
```

**Note:** The `face_recognition` library depends on `dlib`, which can require additional build tools and compatible Python versions, particularly on Windows.

## Installation

1. Download or clone this repository.
2. Install the required Python libraries.
3. Create a folder named `models` in the project's root directory.
4. Download all four required pretrained model files and place them in `models/` (see below).
5. Open a terminal in the project's root directory.
6. Run:

   ```bash
   python MainProgram.py
   ```

### Pretrained Models

**The `models/` folder and its contents are not included in this repository.** You need to create the folder yourself and download the four required files before running the program:

```text
models/
├── age_deploy.prototxt
├── age_net.caffemodel
├── gender_deploy.prototxt
└── gender_net.caffemodel
```

The `.prototxt` files describe the network architectures, while the `.caffemodel` files contain pretrained network weights.

All four model files must be obtained separately. You can find the pretrained models and related files through the original [AgeGenderDeepLearning project by Gil Levi and Tal Hassner](https://github.com/GilLevi/AgeGenderDeepLearning).

Place the files in the `models/` folder you created, keeping the exact filenames shown above. The program expects these paths when run from the repository root.

## How to Use

### 1. Select a Reference Image

When the program starts, provide the path to a photograph containing the face you want to search for.

Example:

```text
C:/Users/YourName/Pictures/reference.jpg
```

The program detects faces in the image and creates numerical encodings.

**Important:** The current implementation uses the first detected face in the reference image. For predictable results, use a reference photograph containing only one clearly visible face.

### 2. Select the Search Folder

Enter the path to the folder containing photographs you want to analyze.

Example:

```text
C:/Users/YourName/Pictures/Photos
```

The program examines supported images directly inside the selected folder.

### 3. Face Comparison

For each photograph, the program:

1. Loads the image.
2. Detects visible faces.
3. Generates a face encoding for each detected face.
4. Compares each encoding against the reference encoding.
5. Processes faces whose calculated distance is below `0.6`.

A lower face distance generally indicates greater similarity between the two encodings.

The `0.6` threshold is a matching heuristic, not a percentage of certainty.

### 4. Age and Gender-Category Prediction

When a face passes the matching threshold, the program crops the face and processes it using two pretrained Caffe neural networks.

The age model predicts one of eight age ranges:

| Category | Predicted age range |
|---|---|
| 1 | 0–2 |
| 2 | 4–6 |
| 3 | 8–12 |
| 4 | 15–20 |
| 5 | 25–32 |
| 6 | 38–43 |
| 7 | 48–53 |
| 8 | 60–100 |

The gender classification model produces one of two categories:

- Male
- Female

**These values are estimates generated from visual features.** The age ranges are approximate, and the binary gender-classifier output must not be treated as proof of a person's actual gender identity.

The program counts the predicted gender categories across all matched faces and selects the majority category for its output labels. If the counts are equal, it uses `Undefined`.

### 5. Saving the Results

Matching photographs are saved into:

```text
Documents/
└── FaceComperesment/
    └── matches/
```

The folder is automatically created when required.

For every matching face, the program draws:

- A green rectangle around the detected face.
- A text label containing the estimated age group and selected gender category.

The resulting image is saved using the original photograph's filename.

The original source photograph is not modified.

## How It Works

The application is divided into multiple Python modules, each responsible for a specific part of the processing pipeline.

### 1. MainProgram.py — Main Application

Controls the main execution sequence.

It handles reference image selection, search folder selection, face encoding generation, and starting the matching algorithm.

### 2. face_processing — Face Detection and Encoding

This folder contains three modules:

**formatinImage.py**

Uses:

```python
face_recognition.load_image_file(image_path)
```

to load an image into a format suitable for face recognition.

**FaceLocation.py**

Calls:

```python
face_recognition.face_locations(formated_image)
```

to identify face locations within the image.

**VectorCreate.py**

Generates facial encodings:

```python
face_recognition.face_encodings(image_format, face_locations)
```

Each detected face is represented by a 128-dimensional vector describing numerical facial characteristics.

### 3. face_algoritm — Face Similarity Comparison

The main matching algorithm is implemented in:

`Algoritm_for_face_SImilarity.py`

It compares the reference face encoding against the encodings of faces found in the search images.

```python
distance = face_recognition.face_distance([org_image_vector], image_vector[x])[0]
```

A potential match is accepted when:

```python
if distance < 0.6:
```

When a matching face is found, the program stores its location and performs age and gender-category predictions.

After processing all photographs, it sends the results to the visualization module.

### 4. age_prediction — Age Classification

The `age_algoritm.py` module loads a pretrained Caffe neural network through OpenCV.

```python
age_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
```

The cropped face is converted into an input blob and passed to the neural network.

The model produces output scores for eight age categories.

The program selects the category with the highest output score using `argmax()`.

### 5. gender_prediction — Gender-Category Classification

The `gender_algoritm.py` module uses another pretrained Caffe neural network.

It processes a cropped face and selects the higher-scoring category from the model's Male and Female outputs.

These predictions are subsequently aggregated by the matching algorithm.

### 6. mapping — File Discovery and Folder Creation

**ImageGetter.py**

Searches the selected directory for supported image filenames:

- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`

**Newfile_creation.py**

Creates the output folders in the user's Windows Documents directory when required.

### 7. post_visual — Result Annotation

The `rectangle_face.py` module uses OpenCV to load a matching photograph and draw a rectangle around the matched face.

```python
cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 3)
```

It also uses `cv2.putText()` to add the predicted age and gender-category labels.

The resulting image is saved using `cv2.imwrite()`.

## Project Structure

```text
Face-Matching-Analyzer/
├── MainProgram.py
│
├── face_processing/
│   ├── FaceLocation.py
│   ├── formatinImage.py
│   └── VectorCreate.py
│
├── face_algoritm/
│   └── Algoritm_for_face_SImilarity.py
│
├── age_prediction/
│   └── age_algoritm.py
│
├── gender_prediction/
│   └── gender_algoritm.py
│
├── mapping/
│   ├── ImageGetter.py
│   └── Newfile_creation.py
│
├── post_visual/
│   └── rectangle_face.py
│
├── .gitignore
└── README.md
```

The `models/` folder is created locally during setup and is not part of the GitHub repository. All four required model files must be downloaded separately before running the program.

## Known Limitations

- **Face recognition accuracy:** The matching algorithm can produce false positives or fail to recognize matching faces under different lighting, angles, or image quality.
- **Fixed threshold:** The face-distance threshold is fixed at `0.6` and cannot be changed through the program interface.
- **Single reference face:** Only the first detected face in the reference photograph is used.
- **Approximate age predictions:** The model predicts predefined age categories rather than exact ages.
- **Binary gender classifier:** The model has only Male and Female output categories and cannot reliably determine gender identity.
- **Shared gender label:** The program uses the majority gender prediction across matching faces rather than independently labeling each match.
- **Multiple matches in one image:** When more than one matching face appears in a photograph, output files may be overwritten because the original filename is reused.
- **No recursive scanning:** Photographs inside subfolders are not processed.
- **Limited image formats:** The search only recognizes lowercase PNG, JPG, JPEG, and BMP filename extensions.
- **Model performance:** Age and gender models are loaded separately for each matching face, which can slow processing.
- **Input preprocessing:** The age model receives a 255 × 255 input blob even though the supplied model architecture specifies 227 × 227 inputs; prediction quality has not been validated.
- **Windows-specific output:** The destination folder uses the Windows `USERPROFILE` environment variable.
- **No graphical interface:** The application operates through the terminal.
- **Limited error handling:** Missing models, invalid images, or unexpected face crops can interrupt processing.

## Development Note

I created this project to explore more advanced computer vision concepts after working on simpler Python image-processing applications.

The program combines facial detection and comparison with pretrained convolutional neural networks, and it organizes the processing pipeline across several modules.

It demonstrates my exploration of face encodings, numerical similarity measurements, image cropping, neural network inference, filesystem operations, and automated image annotation.

The face recognition functionality uses the `face_recognition` library, while age and gender-category predictions are produced by external pretrained models rather than neural networks trained by me.

This is an educational and experimental project, not a validated biometric identification system.

## Third-Party Models

The pretrained age and gender classification models originate from research by Gil Levi and Tal Hassner.

**Reference:**

Gil Levi and Tal Hassner, *Age and Gender Classification Using Convolutional Neural Networks*, IEEE Conference on Computer Vision and Pattern Recognition Workshops, 2015.

[Original research and model repository](https://github.com/GilLevi/AgeGenderDeepLearning)

**Author:** [Nik Berginc](https://github.com/nikberginc)
