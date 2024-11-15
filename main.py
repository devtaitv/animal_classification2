import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import threading
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import torch.nn.functional as F
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json


class ImprovedCNNModel(nn.Module):
    def __init__(self):
        super(ImprovedCNNModel, self).__init__()
        # Improved CNN architecture with batch normalization and dropout
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 128 * 8 * 8)
        x = self.dropout(F.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))
        return x


class PetClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pet Classifier - SVM, ANN, CNN")
        self.root.geometry("1400x800")
        self.root.configure(bg='#f0f0f0')

        # Initialize variables
        self.img_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.initialize_models()
        self.create_gui()

    def initialize_models(self):
        # CNN Model
        self.cnn_model = ImprovedCNNModel().to(self.device)

        # SVM Model
        self.svm_model = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')
        self.scaler = StandardScaler()

        # ANN Model
        self.ann_model = MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation='relu',
            solver='adam',
            max_iter=500,
            learning_rate='adaptive',
            early_stopping=True,
            validation_fraction=0.1
        )

    def load_image(self):
        """Load and display an image for prediction"""
        try:
            # Open file dialog for image selection
            file_path = filedialog.askopenfilename(
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                    ("All files", "*.*")
                ]
            )

            if not file_path:
                return

            # Save the current image path
            self.current_image = file_path

            # Open and resize image for display
            image = Image.open(file_path)
            # Calculate new size while maintaining aspect ratio
            display_size = (300, 300)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)

            # Convert to PhotoImage for display
            self.photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=self.photo)

            # Clear previous prediction results
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Image loaded successfully.\nClick 'Predict' to classify.")

        except Exception as e:
            messagebox.showerror("Error", f"Error loading image: {str(e)}")

    def save_model(self):
        """Save the current model"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".pth",
                filetypes=[
                    ("PyTorch model", "*.pth"),
                    ("All files", "*.*")
                ]
            )
            if file_path:
                if self.model_var.get() == "CNN":
                    torch.save(self.cnn_model.state_dict(), file_path)
                elif self.model_var.get() == "SVM":
                    import joblib
                    joblib.dump(self.svm_model, file_path)
                else:  # ANN
                    import joblib
                    joblib.dump(self.ann_model, file_path)
                messagebox.showinfo("Success", "Model saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving model: {str(e)}")

    def load_model(self):
        """Load a previously saved model"""
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[
                    ("Model files", "*.pth *.joblib"),
                    ("All files", "*.*")
                ]
            )
            if file_path:
                if self.model_var.get() == "CNN":
                    self.cnn_model.load_state_dict(torch.load(file_path))
                elif self.model_var.get() == "SVM":
                    import joblib
                    self.svm_model = joblib.load(file_path)
                else:  # ANN
                    import joblib
                    self.ann_model = joblib.load(file_path)
                messagebox.showinfo("Success", "Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model: {str(e)}")

    def create_gui(self):
        # Main container
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Left panel - Training Controls
        left_panel = ttk.LabelFrame(self.main_frame, text="Training Control", padding="10")
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)

        # Data directory selection
        ttk.Label(left_panel, text="Data Directory:").grid(row=0, column=0, sticky=tk.W)
        self.data_dir_var = tk.StringVar()
        ttk.Entry(left_panel, textvariable=self.data_dir_var, width=30).grid(row=0, column=1, padx=5)
        ttk.Button(left_panel, text="Browse", command=self.browse_directory).grid(row=0, column=2)

        # Model selection
        ttk.Label(left_panel, text="Model:").grid(row=1, column=0, sticky=tk.W, pady=10)
        self.model_var = tk.StringVar(value="CNN")
        ttk.Radiobutton(left_panel, text="CNN", variable=self.model_var, value="CNN").grid(row=1, column=1, sticky=tk.W)
        ttk.Radiobutton(left_panel, text="SVM", variable=self.model_var, value="SVM").grid(row=1, column=2, sticky=tk.W)
        ttk.Radiobutton(left_panel, text="ANN", variable=self.model_var, value="ANN").grid(row=1, column=3, sticky=tk.W)

        # Training parameters
        params_frame = ttk.LabelFrame(left_panel, text="Training Parameters", padding="5")
        params_frame.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=10)

        ttk.Label(params_frame, text="Epochs:").grid(row=0, column=0, sticky=tk.W)
        self.epochs_var = tk.StringVar(value="10")
        ttk.Entry(params_frame, textvariable=self.epochs_var, width=10).grid(row=0, column=1, padx=5)

        ttk.Label(params_frame, text="Batch Size:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.batch_size_var = tk.StringVar(value="32")
        ttk.Entry(params_frame, textvariable=self.batch_size_var, width=10).grid(row=0, column=3, padx=5)

        # Training buttons
        btn_frame = ttk.Frame(left_panel)
        btn_frame.grid(row=3, column=0, columnspan=4, pady=10)

        self.train_btn = ttk.Button(btn_frame, text="Train Model", command=self.start_training)
        self.train_btn.grid(row=0, column=0, padx=5)

        self.stop_btn = ttk.Button(btn_frame, text="Stop Training", command=self.stop_training, state='disabled')
        self.stop_btn.grid(row=0, column=1, padx=5)

        self.save_btn = ttk.Button(btn_frame, text="Save Model", command=self.save_model)
        self.save_btn.grid(row=0, column=2, padx=5)

        self.load_btn = ttk.Button(btn_frame, text="Load Model", command=self.load_model)
        self.load_btn.grid(row=0, column=3, padx=5)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(left_panel, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=4, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=5)

        # Training metrics
        metrics_frame = ttk.LabelFrame(left_panel, text="Training Metrics", padding="5")
        metrics_frame.grid(row=5, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=10)

        self.metrics_text = tk.Text(metrics_frame, height=10, width=50)
        self.metrics_text.grid(row=0, column=0, pady=5)

        # Right panel - Image display and prediction
        right_panel = ttk.LabelFrame(self.main_frame, text="Prediction", padding="10")
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)

        self.image_label = ttk.Label(right_panel)
        self.image_label.grid(row=0, column=0, columnspan=2, pady=10)

        ttk.Button(right_panel, text="Load Image", command=self.load_image).grid(row=1, column=0, pady=5)
        ttk.Button(right_panel, text="Predict", command=self.predict).grid(row=1, column=1, pady=5)

        # Results display
        self.result_text = tk.Text(right_panel, height=5, width=40)
        self.result_text.grid(row=2, column=0, columnspan=2, pady=10)
        # Add plot canvas
        self.figure, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.main_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=2, pady=10)

    def browse_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.data_dir_var.set(directory)

    def preprocess_data(self):
        # Enhanced data preprocessing with data augmentation for training
        train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Load and prepare datasets
        train_dir = os.path.join(self.data_dir_var.get(), 'Train')
        val_dir = os.path.join(self.data_dir_var.get(), 'Validation')

        train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)

        self.class_names = train_dataset.classes

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=int(self.batch_size_var.get()),
            shuffle=True,
            num_workers=2
        )

        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=int(self.batch_size_var.get()),
            shuffle=False,
            num_workers=2
        )

    def start_training(self):
        """Start the training process in a separate thread"""
        if not self.data_dir_var.get():
            messagebox.showerror("Error", "Please select data directory first!")
            return

        # Disable training button and enable stop button
        self.train_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.stop_training_flag = False

        # Clear metrics display and history
        self.metrics_text.delete(1.0, tk.END)
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        # Clear plots
        self.ax1.clear()
        self.ax2.clear()
        self.canvas.draw()

        # Start training in a separate thread
        self.training_thread = threading.Thread(target=self.train_model)
        self.training_thread.start()

    def train_model(self):
        """Main training function that handles different models"""
        try:
            self.preprocess_data()

            if self.model_var.get() == "CNN":
                self.train_cnn()
            elif self.model_var.get() == "SVM":
                self.train_svm()
            else:  # ANN
                self.train_ann()

            if not hasattr(self, 'stop_training_flag') or not self.stop_training_flag:
                messagebox.showinfo("Training Completed",
                                    f"{self.model_var.get()} training completed successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during training: {str(e)}")
        finally:
            self.train_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
            self.progress_var.set(0)

    def stop_training(self):
        """Stop the training process"""
        self.stop_training_flag = True
        messagebox.showinfo("Training Stopped", "Training process will stop after current epoch")
        self.train_btn.config(state='normal')
        self.stop_btn.config(state='disabled')

    def train_svm(self):
        """Train the SVM model with progress updates"""
        try:
            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.insert(tk.END, "Starting SVM training...\n")
            self.root.update_idletasks()

            # Initialize lists to store data
            X_train = []
            y_train = []
            X_val = []
            y_val = []

            # Process training data with progress updates
            self.metrics_text.insert(tk.END, "Loading training data...\n")
            total_train_batches = len(self.train_loader)
            for idx, (inputs, labels) in enumerate(self.train_loader):
                if hasattr(self, 'stop_training_flag') and self.stop_training_flag:
                    return

                inputs = inputs.view(inputs.size(0), -1).numpy()
                X_train.extend(inputs)
                y_train.extend(labels.numpy())

                # Update progress
                progress = (idx + 1) / total_train_batches * 40  # First 40% for loading train data
                self.progress_var.set(progress)
                self.metrics_text.insert(tk.END, f"\rProcessing training batch {idx + 1}/{total_train_batches}")
                self.metrics_text.see(tk.END)
                self.root.update_idletasks()

            # Process validation data with progress updates
            self.metrics_text.insert(tk.END, "\nLoading validation data...\n")
            total_val_batches = len(self.val_loader)
            for idx, (inputs, labels) in enumerate(self.val_loader):
                if hasattr(self, 'stop_training_flag') and self.stop_training_flag:
                    return

                inputs = inputs.view(inputs.size(0), -1).numpy()
                X_val.extend(inputs)
                y_val.extend(labels.numpy())

                # Update progress
                progress = 40 + (idx + 1) / total_val_batches * 20  # Next 20% for loading val data
                self.progress_var.set(progress)
                self.metrics_text.insert(tk.END, f"\rProcessing validation batch {idx + 1}/{total_val_batches}")
                self.metrics_text.see(tk.END)
                self.root.update_idletasks()

            # Convert to numpy arrays
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            X_val = np.array(X_val)
            y_val = np.array(y_val)

            # Scale the data
            self.metrics_text.insert(tk.END, "\n\nScaling data...\n")
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            self.progress_var.set(65)  # 65% progress after scaling
            self.root.update_idletasks()

            # Train SVM with progress updates
            self.metrics_text.insert(tk.END, "Training SVM model...\n")
            # Use smaller chunks for partial_fit if available, otherwise use regular fit
            chunk_size = 1000
            total_chunks = (len(X_train) + chunk_size - 1) // chunk_size

            if hasattr(self.svm_model, 'partial_fit'):
                for i in range(0, len(X_train), chunk_size):
                    if hasattr(self, 'stop_training_flag') and self.stop_training_flag:
                        return

                    end_idx = min(i + chunk_size, len(X_train))
                    self.svm_model.partial_fit(
                        X_train[i:end_idx],
                        y_train[i:end_idx],
                        classes=np.unique(y_train)
                    )

                    # Update progress
                    chunk_progress = (i + chunk_size) / len(X_train)
                    progress = 65 + chunk_progress * 25  # Last 25% for training
                    self.progress_var.set(progress)
                    self.metrics_text.insert(tk.END, f"\rTraining progress: {progress:.1f}%")
                    self.metrics_text.see(tk.END)
                    self.root.update_idletasks()
            else:
                self.metrics_text.insert(tk.END, "Fitting SVM model (this may take a while)...\n")
                self.svm_model.fit(X_train, y_train)
                self.progress_var.set(90)
                self.root.update_idletasks()

            # Calculate and display metrics
            self.metrics_text.insert(tk.END, "\n\nCalculating final metrics...\n")

            # Training metrics
            train_pred = self.svm_model.predict(X_train)
            train_accuracy = accuracy_score(y_train, train_pred) * 100

            # Validation metrics
            val_pred = self.svm_model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred) * 100

            # Display final results
            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.insert(tk.END, f"Training completed!\n\n")
            self.metrics_text.insert(tk.END, f"SVM Training Accuracy: {train_accuracy:.2f}%\n")
            self.metrics_text.insert(tk.END, f"SVM Validation Accuracy: {val_accuracy:.2f}%\n")
            self.metrics_text.insert(tk.END, "\nClassification Report:\n")
            self.metrics_text.insert(tk.END, classification_report(y_val, val_pred))

            # Update plots
            self.history['train_acc'].append(train_accuracy)
            self.history['val_acc'].append(val_accuracy)
            self.history['train_loss'].append(0)  # SVM doesn't have loss
            self.history['val_loss'].append(0)  # SVM doesn't have loss
            self.plot_training_progress()

            # Final progress update
            self.progress_var.set(100)
            self.root.update_idletasks()

        except Exception as e:
            self.metrics_text.insert(tk.END, f"\nError during SVM training: {str(e)}\n")
            raise e

    def update_training_progress(self, progress, message):
        """Helper method to update training progress"""
        self.progress_var.set(progress)
        if message:
            self.metrics_text.insert(tk.END, f"\n{message}")
            self.metrics_text.see(tk.END)
        self.root.update_idletasks()

    def train_ann(self):
        """Train the ANN model"""
        # Prepare data for ANN
        X_train = []
        y_train = []
        X_val = []
        y_val = []

        # Process training data
        for inputs, labels in self.train_loader:
            inputs = inputs.view(inputs.size(0), -1).numpy()
            X_train.extend(inputs)
            y_train.extend(labels.numpy())

        # Process validation data
        for inputs, labels in self.val_loader:
            inputs = inputs.view(inputs.size(0), -1).numpy()
            X_val.extend(inputs)
            y_val.extend(labels.numpy())

        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)

        # Scale the data
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        # Train ANN
        self.ann_model.fit(X_train, y_train)

        # Calculate accuracies
        train_pred = self.ann_model.predict(X_train)
        val_pred = self.ann_model.predict(X_val)

        train_accuracy = accuracy_score(y_train, train_pred) * 100
        val_accuracy = accuracy_score(y_val, val_pred) * 100

        # Update metrics display
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(tk.END, f"ANN Training Accuracy: {train_accuracy:.2f}%\n")
        self.metrics_text.insert(tk.END, f"ANN Validation Accuracy: {val_accuracy:.2f}%\n")
        self.metrics_text.insert(tk.END, "\nClassification Report:\n")
        self.metrics_text.insert(tk.END, classification_report(y_val, val_pred))

        # Update progress bar
        self.progress_var.set(100)

    def update_metrics(self, epoch, loss, train_acc, val_acc):
        """Update the metrics display"""
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(tk.END, f"Epoch: {epoch}\n")
        self.metrics_text.insert(tk.END, f"Loss: {loss:.4f}\n")
        self.metrics_text.insert(tk.END, f"Training Accuracy: {train_acc:.2f}%\n")
        self.metrics_text.insert(tk.END, f"Validation Accuracy: {val_acc:.2f}%\n")
        self.metrics_text.see(tk.END)

    def train_cnn(self):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.cnn_model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

        epochs = int(self.epochs_var.get())
        best_val_acc = 0

        for epoch in range(epochs):
            if hasattr(self, 'stop_training_flag') and self.stop_training_flag:
                break

            # Training phase
            self.cnn_model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).float()

                optimizer.zero_grad()
                outputs = self.cnn_model(inputs).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            # Validation phase
            self.cnn_model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device).float()

                    outputs = self.cnn_model(inputs).squeeze()
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            # Calculate metrics
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            train_loss = train_loss / len(self.train_loader)
            val_loss = val_loss / len(self.val_loader)

            # Update learning rate
            scheduler.step(val_loss)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.cnn_model.state_dict(), 'best_model.pth')

            # Update history and plots
            self.update_history(train_loss, train_acc, val_loss, val_acc)
            self.plot_training_progress()

            # Update metrics display
            self.update_metrics(epoch + 1, train_loss, train_acc, val_acc)

            # Update progress bar
            progress = (epoch + 1) / epochs * 100
            self.progress_var.set(progress)
            self.root.update_idletasks()

    def predict(self):
        if not hasattr(self, 'current_image'):
            messagebox.showerror("Error", "Please load an image first!")
            return

        try:
            # Load and preprocess image
            processed_image = self.preprocess_single_image(self.current_image)

            prediction = None
            confidence = None

            if self.model_var.get() == "CNN":
                self.cnn_model.eval()
                with torch.no_grad():
                    prediction = self.cnn_model(processed_image).item()
                    label = self.class_names[1] if prediction > 0.5 else self.class_names[0]
                    confidence = max(prediction, 1 - prediction) * 100

            elif self.model_var.get() == "SVM":
                flattened_image = processed_image.view(-1).numpy().reshape(1, -1)
                scaled_image = self.scaler.transform(flattened_image)
                prediction = self.svm_model.predict(scaled_image)[0]
                proba = self.svm_model.predict_proba(scaled_image)[0]
                label = self.class_names[prediction]
                confidence = max(proba) * 100

            else:  # ANN
                flattened_image = processed_image.view(-1).numpy().reshape(1, -1)
                scaled_image = self.scaler.transform(flattened_image)
                prediction = self.ann_model.predict(scaled_image)[0]
                proba = self.ann_model.predict_proba(scaled_image)[0]
                label = self.class_names[prediction]
                confidence = max(proba) * 100

            # Display results
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Prediction: {label}\n")
            self.result_text.insert(tk.END, f"Confidence: {confidence:.2f}%\n")

            # Save prediction history
            self.save_prediction({
                'image_path': self.current_image,
                'prediction': label,
                'confidence': confidence,
                'model': self.model_var.get()
            })

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during prediction: {str(e)}")

    def save_prediction(self, prediction_data):
        """Save prediction history to a JSON file"""
        history_file = 'prediction_history.json'
        history = []

        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)

        history.append(prediction_data)

        with open(history_file, 'w') as f:
            json.dump(history, f, indent=4)

    def update_history(self, train_loss, train_acc, val_loss, val_acc):
        """Update training history"""
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)

    def plot_training_progress(self):
        """Update training progress plots"""
        self.ax1.clear()
        self.ax2.clear()

        # Plot accuracy
        self.ax1.plot(self.history['train_acc'], label='Train')
        self.ax1.plot(self.history['val_acc'], label='Validation')
        self.ax1.set_title('Accuracy')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Accuracy (%)')
        self.ax1.legend()

        # Plot loss
        self.ax2.plot(self.history['train_loss'], label='Train')
        self.ax2.plot(self.history['val_loss'], label='Validation')
        self.ax2.set_title('Loss')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Loss')
        self.ax2.legend()

        self.figure.tight_layout()
        self.canvas.draw()

    def preprocess_single_image(self, image_path):
        """Preprocess a single image for prediction"""
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image)

        if self.model_var.get() == "CNN":
            return image_tensor.unsqueeze(0).to(self.device)
        else:
            return image_tensor

    def show_training_history(self):
        """Display training history in a new window"""
        history_window = tk.Toplevel(self.root)
        history_window.title("Training History")
        history_window.geometry("600x400")

        # Create text widget to display history
        history_text = tk.Text(history_window, wrap=tk.WORD, padx=10, pady=10)
        history_text.pack(fill=tk.BOTH, expand=True)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(history_window, command=history_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        history_text.configure(yscrollcommand=scrollbar.set)

        # Load and display history
        try:
            with open('training_history.json', 'r') as f:
                history = json.load(f)
                history_text.insert(tk.END, json.dumps(history, indent=4))
        except FileNotFoundError:
            history_text.insert(tk.END, "No training history found.")
        except Exception as e:
            history_text.insert(tk.END, f"Error loading history: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PetClassifierApp(root)
    root.mainloop()