# AI - Game / Generate Procedural Maps

## 🛠 Tech Stack
- **C (Language)** - Core game logic and ONNX Runtime integration
- **Python (Language)** - Training AI model
- **PyTorch (AI Framework)** - Training deep learning model
- **ONNX (Model Format)** - Portable model format for inference in C
- **ONNX Runtime (Inference Engine)** - Running ONNX models efficiently in C
- **Pygame (Game Framework)** - Displaying AI-generated maps for debugging
- **CMake (Build System)** - Compiling the C integration

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
│   ├── CMakelists.txt                 # Compilation instructions
├── 📜 .gitignore                      # Git ignore rules
├── 📜 requirements.txt                # Python dependencies
├── 📜 README.md                       # Project documentation
```

<br /><br />

## 🚀 Overview
Ce projet vise à **générer des cartes procédurales** pour un jeu similaire à « King Arthur's Gold » en utilisant un réseau neuronal entraîné en Python. Le modèle est exporté vers **ONNX** et exécuté en **C** via **ONNX Runtime** pour la génération de cartes en temps réel.

<br /><br />

## 🔥 Why ONNX ?
ONNX (Open Neural Network Exchange) nous permet de :
- Former un modèle d'IA en Python et l'utiliser en C **sans avoir besoin de Python à l'exécution**.
- Optimiser pour de multiples plateformes (**Windows, Linux, macOS, iOS, Android**).
- Accélérer l'inférence avec des **optimisations GPU (CUDA, TensorRT, DirectML, Metal)**.

<br /><br />

## ⚙️ Setup Environment - Development

### 1️⃣ Download and Install Python >= 3.12.3
URL : https://www.python.org/downloads/

### 2️⃣ Activer le support des chemins longs (Windows)
```bash
# Ouvrir PowerShell en mode administrateur
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1

# Redémarer sont terminal ensuite
```

### 3️⃣ Créer un environnement virtuel propre (Windows)
```bash
# Créer l'environnement virtuel
python -m venv C:\venv\myproject_env

# Activer l'environnement virtuel
C:\venv\myproject_env\Scripts\activate
```

### 4️⃣ Visual Studio + CMake + zlib for ONNX (Windows)
1. Visual Studio: Download and install Visual Studio 2022 with MSVC >= v143, Windows SDK >= 10 and component C/C++ desktop in Visual Studio.
2. CMake: Ensure CMake 3.25 or newer is installed and added to your PATH.
3. zlib:
```bash
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg install zlib:x64-windows
.\vcpkg integrate install
```

### 5️⃣ Install Python dependencies
```bash
pip install -r requirements.txt 
```

<br /><br />

## 📌 Usage

### 1️⃣ Train and export the AI model
```bash
C:\venv\myproject_env\Scripts\activate
cd src/ai_model
python train.py
```
This will generate `map_generator.onnx` inside the `models/` folder.

### 2️⃣ Run visualization with Pygame
```bash
C:\venv\myproject_env\Scripts\activate
cd src/visualization
python display.py
```
Press **SPACE** to generate new maps in real-time.

### 3️⃣ Compile and run the C integration
```bash
cd game
# ... TODO
```