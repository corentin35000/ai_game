# AI - Game / Generate Procedural Maps

## 🛠 Tech Stack
- **C (Language)** - Core game logic and ONNX Runtime integration
- **Python (Language)** - Training AI model
- **PyTorch (AI Framework)** - Training deep learning model
- **ONNX (Model Format)** - Portable model format for inference in C
- **ONNX Runtime (Inference Engine)** - Running ONNX models efficiently in C
- **Pygame (Visualization)** - Displaying AI-generated maps for debugging
- **Makefile (Build System)** - Compiling the C integration

<br /><br />

## 📂 Project Structure

This project follows a structured layout to keep the **AI model, training, visualization, and integration with C** well-organized.

```
📦 root-project/
├── 📂 src/                          # Source directory
│   ├── 📂 ai_model/                 # AI model training and export
│   │   ├── model.py                 # Definition of the neural network model
│   │   ├── train.py                 # Training and exporting to ONNX
│   │   ├── utils.py                  # Utility functions (normalization, preprocessing, etc.)
│   ├── 📂 visualization/             # Pygame visualization of generated maps
│   │   ├── display.py                # Rendering maps in real-time with Pygame
├── 📂 models/                        # Trained models stored here
│   ├── map_generator.onnx            # Exported ONNX model for inference in C
├── 📂 game/                          # Integration of ONNX Runtime in C
│   ├── main.c                         # Loading ONNX model in C and generating maps
│   ├── Makefile                      # Compilation instructions
├── 📜 .gitignore                      # Git ignore rules
├── 📜 requirements.txt                # Python dependencies
├── 📜 README.md                       # Project documentation
```

<br /><br />

## 🚀 Overview
This project aims to **generate procedural maps** for a game similar to "King Arthur’s Gold" using a neural network trained in Python. The model is exported to **ONNX** and executed in **C** via **ONNX Runtime** for real-time map generation.

<br /><br />

### 🔥 Why ONNX?
ONNX (Open Neural Network Exchange) allows us to:
- Train an AI model in Python and use it in C **without requiring Python at runtime**.
- Optimize for multiple platforms (**Windows, Linux, macOS, iOS, Android**).
- Accelerate inference with **GPU optimizations (CUDA, TensorRT, DirectML, Metal)**.

<br /><br />

## ⚙️ Setup Environment - Development

### 1️⃣ Install dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt 
```

## 📌 Usage

### 1️⃣ Train and export the AI model
```bash
cd src/ai_model
python train.py
```
This will generate `map_generator.onnx` inside the `models/` folder.

### 2️⃣ Run visualization with Pygame
```bash
cd src/visualization
python display.py
```
Press **SPACE** to generate new maps in real-time.

### 3️⃣ Compile and run the C integration
```bash
cd game
make    # Compile main.c with ONNX Runtime
./main  # Run the ONNX inference in C
```