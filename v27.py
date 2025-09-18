#!/usr/bin/env python3
"""
Real-Time Emotion and Identity Recognition System
With Valence-Arousal Graph Implementation
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time
import json
import os
from datetime import datetime
from collections import defaultdict, deque
from PIL import Image, ImageTk
import subprocess
import platform

# Try importing DeepFace with better error handling
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("DeepFace imported successfully")
except ImportError as e:
    print(f"DeepFace import failed: {e}")
    DEEPFACE_AVAILABLE = False

class EmotionIdentityRecognizer:
    """
    Improved Emotion and Identity Recognition System with Valence-Arousal Graph
    """
    
    def __init__(self):
        """Initialize the system"""
        self.root = tk.Tk()
        self.root.title("Emotion & Identity Recognition System")
        self.root.geometry("1600x900")  # Increased width for valence-arousal graph
        
        # Check DeepFace availability
        if not DEEPFACE_AVAILABLE:
            messagebox.showerror("Error", 
                "DeepFace not available. Please install with:\npip install deepface")
            return
        
        # Video capture
        self.cap = None
        self.is_running = False
        
        # Data storage
        self.local_images_db = "local_images"
        self.local_face_data = {}  # Store person name and image path
        self.emotion_history = defaultdict(lambda: deque(maxlen=50))
        self.mood_history = defaultdict(lambda: deque(maxlen=50))  # Store mood scores
        
        # Real-time data
        self.current_frame = None
        self.detected_faces = []
        self.analysis_active = False
        
        # Mood scoring weights (positive vs negative emotions)
        self.positive_emotions = {'happy': 1.0, 'surprise': 0.3}
        self.negative_emotions = {'sad': -1.0, 'angry': -0.9, 'fear': -0.7, 'disgust': -0.8}
        self.neutral_emotions = {'neutral': 0.0}
        
        # Valence-Arousal mapping for emotions
        self.emotion_valence_arousal = {
            'happy': (0.7, 0.6),      # Positive valence, moderate-high arousal
            'sad': (-0.6, -0.3),      # Negative valence, low arousal
            'angry': (-0.8, 0.8),     # Negative valence, high arousal
            'fear': (-0.5, 0.7),      # Negative valence, high arousal
            'surprise': (0.1, 0.8),   # Slightly positive valence, high arousal
            'disgust': (-0.7, 0.2),   # Negative valence, low-moderate arousal
            'neutral': (0.0, 0.0)     # Neutral valence and arousal
        }
        
        # GUI components
        self.video_label = None
        self.info_text = None
        self.mood_figure = None
        self.mood_ax = None
        self.mood_canvas = None
        self.valence_figure = None
        self.valence_ax = None
        self.valence_canvas = None
        
        # Threading
        self.video_thread = None
        self.analysis_thread = None
        
        # Initialize
        self.init_local_database()
        self.setup_gui()
        
        print("System initialized successfully")
    
    def calculate_mood_score(self, emotions):
        """Calculate mood score from -100 (worst) to +100 (best)"""
        if not emotions:
            return 0
        
        mood_score = 0
        total_weight = 0
        
        # Calculate weighted mood score
        for emotion, value in emotions.items():
            if emotion in self.positive_emotions:
                weight = self.positive_emotions[emotion]
                mood_score += value * weight
                total_weight += abs(weight) * value
            elif emotion in self.negative_emotions:
                weight = self.negative_emotions[emotion]
                mood_score += value * weight
                total_weight += abs(weight) * value
            elif emotion in self.neutral_emotions:
                # Neutral emotions don't affect mood much but add to total weight
                total_weight += value * 0.1
        
        # Normalize to -100 to +100 range
        if total_weight > 0:
            normalized_score = (mood_score / total_weight) * 100
            # Clamp to range
            return max(-100, min(100, normalized_score))
        
        return 0
    
    def calculate_valence_arousal(self, emotions):
        """Calculate valence and arousal values from emotion probabilities"""
        if not emotions:
            return 0.0, 0.0
        
        valence = 0.0
        arousal = 0.0
        total_weight = 0.0
        
        # Calculate weighted valence and arousal
        for emotion, probability in emotions.items():
            if emotion in self.emotion_valence_arousal:
                emotion_valence, emotion_arousal = self.emotion_valence_arousal[emotion]
                weight = probability / 100.0  # Convert percentage to weight
                
                valence += emotion_valence * weight
                arousal += emotion_arousal * weight
                total_weight += weight
        
        # Normalize
        if total_weight > 0:
            valence /= total_weight
            arousal /= total_weight
        
        # Clamp to [-1, 1] range
        valence = max(-1.0, min(1.0, valence))
        arousal = max(-1.0, min(1.0, arousal))
        
        return valence, arousal
    
    def init_local_database(self):
        """Initialize local images database"""
        try:
            if not os.path.exists(self.local_images_db):
                os.makedirs(self.local_images_db)
                print(f"Created directory: {self.local_images_db}")
                
                # Create instructions
                readme_path = os.path.join(self.local_images_db, "INSTRUCTIONS.txt")
                with open(readme_path, 'w') as f:
                    f.write("""HOW TO ADD PEOPLE TO RECOGNIZE:

1. Place clear face photos in this folder
2. Name files like: john_doe.jpg, jane_smith.png, alex_johnson.jpeg
3. Use underscores for spaces in names
4. Supported formats: .jpg, .jpeg, .png, .bmp
5. One face per image works best
6. Good lighting and frontal face preferred

Examples:
- john_doe.jpg
- mary_johnson.png
- alex_smith.jpeg

After adding images, click 'Refresh Database' in the app.
""")
            
            self.load_local_faces()
            
        except Exception as e:
            print(f"Error initializing database: {e}")
    
    def load_local_faces(self):
        """Load faces from local images folder"""
        self.local_face_data = {}
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
        
        try:
            if not os.path.exists(self.local_images_db):
                return
                
            for filename in os.listdir(self.local_images_db):
                if filename.lower().endswith(supported_formats):
                    # Extract name from filename
                    person_name = os.path.splitext(filename)[0].replace('_', ' ').title()
                    image_path = os.path.join(self.local_images_db, filename)
                    
                    # Verify image can be loaded
                    try:
                        img = cv2.imread(image_path)
                        if img is not None and img.size > 0:
                            self.local_face_data[person_name] = image_path
                            print(f"Loaded: {person_name} -> {filename}")
                    except Exception as e:
                        print(f"Failed to load {filename}: {e}")
            
            print(f"Total faces loaded: {len(self.local_face_data)}")
            
        except Exception as e:
            print(f"Error loading local faces: {e}")
    #STARTS HERE
    def setup_gui(self):
        """Setup the GUI interface with optimized layout for mood tracking"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Header (10% height)
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.start_btn = ttk.Button(header_frame, text="Start Camera", command=self.start_system)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_btn = ttk.Button(header_frame, text="Stop Camera", command=self.stop_system, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.save_face_btn = ttk.Button(header_frame, text="📸 Save New Face", command=self.save_new_face, state=tk.DISABLED)
        self.save_face_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        refresh_btn = ttk.Button(header_frame, text="Refresh Database", command=self.refresh_database)
        refresh_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        folder_btn = ttk.Button(header_frame, text="Open Images Folder", command=self.open_images_folder)
        folder_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.status_var = tk.StringVar(value="Ready - Add images to local_images folder")
        status_label = ttk.Label(header_frame, textvariable=self.status_var)
        status_label.pack(side=tk.RIGHT)

        # Main content (90% height) - Using grid for better control
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights for proportional sizing - Maximum left panel space
        content_frame.grid_columnconfigure(0, weight=20)  # Left panel - 95.2% width  
        content_frame.grid_columnconfigure(1, weight=1)   # Right panel - 4.8% width
        content_frame.grid_rowconfigure(0, weight=1)

        # Left panel (50% width)
        left_panel = ttk.Frame(content_frame)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        # Configure left panel grid
        left_panel.grid_rowconfigure(0, weight=4)  # Video gets 80% of left panel
        left_panel.grid_rowconfigure(1, weight=1)  # Bottom gets 20% of left panel
        left_panel.grid_columnconfigure(0, weight=1)

        # Video feed (80% height of left panel) - centered
        video_frame = ttk.LabelFrame(left_panel, text="Live Video Feed")
        video_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        
        # Create a frame to center the video
        video_center_frame = ttk.Frame(video_frame)
        video_center_frame.pack(expand=True, fill=tk.BOTH)
        
        self.video_label = ttk.Label(video_center_frame, text="Click 'Start Camera' to begin", anchor="center")
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")

        # Bottom section of left panel (20% height)
        bottom_left_frame = ttk.Frame(left_panel)
        bottom_left_frame.grid(row=1, column=0, sticky="nsew")
        
        # Configure bottom left frame grid - Valence-arousal gets more space for square aspect
        bottom_left_frame.grid_columnconfigure(0, weight=2)  # Detection results - 40%
        bottom_left_frame.grid_columnconfigure(1, weight=3)  # Valence-arousal - 60% (square aspect)
        bottom_left_frame.grid_rowconfigure(0, weight=1)

        # Detection results (REDUCED width - now only 33% of bottom section)
        detection_frame = ttk.LabelFrame(bottom_left_frame, text="Detection")
        detection_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        # Detection results text widget with proper width for single-line "loaded" text
        self.info_text = tk.Text(detection_frame, height=6, width=42, wrap=tk.WORD, font=("Arial", 10))
        scrollbar = ttk.Scrollbar(detection_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scrollbar.set)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Valence-arousal graph (INCREASED width - now 67% of bottom section)
        valence_frame = ttk.LabelFrame(bottom_left_frame, text="Valence-Arousal Graph")
        valence_frame.grid(row=0, column=1, sticky="nsew")
        
        self.valence_figure = Figure(figsize=(5, 4), dpi=80, tight_layout=True)
        self.valence_canvas = FigureCanvasTkAgg(self.valence_figure, valence_frame)
        self.valence_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        self.valence_ax = self.valence_figure.add_subplot(111)
        self.setup_valence_arousal_plot()

        # Right panel (50% width) - INCREASED space for mood tracking
        right_panel = ttk.Frame(content_frame)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        # Mood score tracking (FULL right panel space)
        mood_frame = ttk.LabelFrame(right_panel, text="Mood Score Analysis (Best +100 ↔ Worst -100)")
        mood_frame.pack(fill=tk.BOTH, expand=True)
        
        # Moderately sized mood graph 
        self.mood_figure = Figure(figsize=(8, 7), dpi=80, tight_layout=True)
        self.mood_canvas = FigureCanvasTkAgg(self.mood_figure, mood_frame)
        self.mood_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        self.mood_ax = self.mood_figure.add_subplot(111)
        self.mood_ax.set_title("Waiting for data...")
        self.mood_figure.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.15)
        self.mood_canvas.draw()

        # Initial info update
        self.update_info_display()
        #ENDS HERE
    
    def setup_valence_arousal_plot(self):
        """Setup the valence-arousal plot with emotion labels"""
        try:
            self.valence_ax.clear()
            
            # Set up the plot
            self.valence_ax.set_xlim(-1.1, 1.1)
            self.valence_ax.set_ylim(-1.1, 1.1)
            self.valence_ax.set_xlabel("Valence (Negative ← → Positive)")
            self.valence_ax.set_ylabel("Arousal (Low ← → High)")
            self.valence_ax.set_title("Emotion Valence-Arousal Space")
            
            # Add grid
            self.valence_ax.grid(True, alpha=0.3)
            
            # Add axes lines
            self.valence_ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            self.valence_ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # Add quadrant labels
            self.valence_ax.text(0.7, 0.7, 'High Arousal\nPositive', ha='center', va='center', 
                                fontsize=8, alpha=0.6, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
            self.valence_ax.text(-0.7, 0.7, 'High Arousal\nNegative', ha='center', va='center', 
                                fontsize=8, alpha=0.6, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
            self.valence_ax.text(0.7, -0.7, 'Low Arousal\nPositive', ha='center', va='center', 
                                fontsize=8, alpha=0.6, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
            self.valence_ax.text(-0.7, -0.7, 'Low Arousal\nNegative', ha='center', va='center', 
                                fontsize=8, alpha=0.6, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
            
            # Plot all emotion positions as reference points
            emotion_colors = {
                'happy': 'gold',
                'sad': 'blue',
                'angry': 'red',
                'fear': 'orange',
                'surprise': 'purple',
                'disgust': 'brown',
                'neutral': 'gray'
            }
            
            for emotion, (valence, arousal) in self.emotion_valence_arousal.items():
                color = emotion_colors.get(emotion, 'black')
                self.valence_ax.scatter(valence, arousal, c=color, s=50, alpha=0.4, edgecolors='black', linewidth=0.5)
                self.valence_ax.annotate(emotion.capitalize(), (valence, arousal), 
                                        xytext=(5, 5), textcoords='offset points', 
                                        fontsize=8, alpha=0.7)
            
            # Add text for when no faces are detected
            self.valence_ax.text(0, 0, 'No faces detected\nWaiting for analysis...', 
                                ha='center', va='center', fontsize=10, alpha=0.8,
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            self.valence_canvas.draw()
            
        except Exception as e:
            print(f"Error setting up valence-arousal plot: {e}")
    
    def update_valence_arousal_plot(self):
        """Update the valence-arousal plot with current detections"""
        try:
            self.valence_ax.clear()
            
            # Re-setup the base plot
            self.valence_ax.set_xlim(-1.1, 1.1)
            self.valence_ax.set_ylim(-1.1, 1.1)
            self.valence_ax.set_xlabel("Valence (Negative ← → Positive)")
            self.valence_ax.set_ylabel("Arousal (Low ← → High)")
            self.valence_ax.set_title("Current Emotion Positions")
            
            # Add grid and axes
            self.valence_ax.grid(True, alpha=0.3)
            self.valence_ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            self.valence_ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # Add quadrant backgrounds
            self.valence_ax.fill_between([0, 1.1], 0, 1.1, alpha=0.05, color='green')  # Positive/High
            self.valence_ax.fill_between([-1.1, 0], 0, 1.1, alpha=0.05, color='red')   # Negative/High
            self.valence_ax.fill_between([0, 1.1], -1.1, 0, alpha=0.05, color='blue')  # Positive/Low
            self.valence_ax.fill_between([-1.1, 0], -1.1, 0, alpha=0.05, color='gray') # Negative/Low
            
            # Plot reference emotion positions (smaller and lighter)
            emotion_colors = {
                'happy': 'gold',
                'sad': 'blue',
                'angry': 'red',
                'fear': 'orange',
                'surprise': 'purple',
                'disgust': 'brown',
                'neutral': 'gray'
            }
            
            for emotion, (valence, arousal) in self.emotion_valence_arousal.items():
                color = emotion_colors.get(emotion, 'black')
                self.valence_ax.scatter(valence, arousal, c=color, s=30, alpha=0.2, 
                                      edgecolors='black', linewidth=0.5)
                self.valence_ax.annotate(emotion, (valence, arousal), 
                                        xytext=(3, 3), textcoords='offset points', 
                                        fontsize=7, alpha=0.4)
            
            # Plot current detections
            if not self.detected_faces:
                self.valence_ax.text(0, 0, 'No faces detected', 
                                    ha='center', va='center', fontsize=12, alpha=0.7,
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                person_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
                
                for i, face_data in enumerate(self.detected_faces):
                    try:
                        emotions = face_data.get('emotions', {})
                        identity = face_data.get('identity', f'Person {i+1}')
                        
                        if emotions:
                            # Calculate valence and arousal
                            valence, arousal = self.calculate_valence_arousal(emotions)
                            
                            # Plot current position
                            color = person_colors[i % len(person_colors)]
                            self.valence_ax.scatter(valence, arousal, c=color, s=200, 
                                                  alpha=0.8, edgecolors='black', linewidth=2,
                                                  marker='o', zorder=5)
                            
                            # Add person label
                            self.valence_ax.annotate(identity, (valence, arousal), 
                                                    xytext=(10, 10), textcoords='offset points', 
                                                    fontsize=10, fontweight='bold',
                                                    bbox=dict(boxstyle='round,pad=0.3', 
                                                            facecolor=color, alpha=0.7))
                            
                            # Add dominant emotion info
                            dominant_emotion = face_data.get('dominant_emotion', 'neutral')
                            emotion_score = emotions.get(dominant_emotion, 0)
                            info_text = f"{dominant_emotion}\n{emotion_score:.1f}%"
                            self.valence_ax.annotate(info_text, (valence, arousal), 
                                                    xytext=(10, -25), textcoords='offset points', 
                                                    fontsize=8, alpha=0.8,
                                                    bbox=dict(boxstyle='round,pad=0.2', 
                                                            facecolor='white', alpha=0.8))
                    
                    except Exception as e:
                        print(f"Error plotting face {i}: {e}")
            
            self.valence_canvas.draw()
            
        except Exception as e:
            print(f"Error updating valence-arousal plot: {e}")
    
    def start_system(self):
        """Start camera and analysis"""
        try:
            # Test camera
            self.cap = cv2.VideoCapture(1) #for external cam access
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Cannot access camera")
                return
            
            self.is_running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_var.set("Running - Analyzing faces...")
            
            self.save_face_btn.config(state=tk.NORMAL)  # Enable save face button
            
            # Start threads
            self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
            self.analysis_thread = threading.Thread(target=self.analysis_loop, daemon=True)
            
            self.video_thread.start()
            self.analysis_thread.start()
            
            print("System started successfully")
            
        except Exception as e:
            print(f"Error starting system: {e}")
            messagebox.showerror("Error", f"Failed to start: {e}")
    
    def stop_system(self):
        """Stop camera and analysis"""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Stopped")
        
        self.save_face_btn.config(state=tk.DISABLED)  # Disable save face button
        
        # Clear video
        self.video_label.config(image="", text="Camera stopped")
        
        # Reset valence-arousal plot
        self.setup_valence_arousal_plot()
        
        print("System stopped")
    
    def video_loop(self):
        """Video capture loop with faster face tracking"""
        frame_count = 0
        while self.is_running and self.cap:
            try:
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame = frame.copy()
                    
                    # Update face rectangles every frame for smooth tracking
                    display_frame = self.add_face_rectangles(frame)
                    
                    # Only update GUI every 2nd frame for better performance
                    if frame_count % 2 == 0:
                        self.update_video_display(display_frame)
                    
                    frame_count += 1
                
                time.sleep(0.016)  # ~60 FPS for smoother tracking
                
            except Exception as e:
                print(f"Video loop error: {e}")
                break
    
    def add_face_rectangles(self, frame):
        """Add rectangles around detected faces"""
        display_frame = frame.copy()
        
        for i, face_data in enumerate(self.detected_faces):
            try:
                region = face_data.get('region', {})
                x = int(region.get('x', 0))
                y = int(region.get('y', 0))
                w = int(region.get('w', 0))
                h = int(region.get('h', 0))
                
                if w > 0 and h > 0:
                    # Draw rectangle
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Add label
                    name = face_data.get('identity', 'Unknown')
                    emotion = face_data.get('dominant_emotion', '')
                    age = face_data.get('age', '')
                    mood_score = face_data.get('mood_score', 0)
                    
                    label1 = f"{name}"
                    label2 = f"{emotion} | Age:{age}"
                    label3 = f"Mood: {mood_score:+.0f}"
                    
                    # Draw labels with background
                    cv2.putText(display_frame, label1, (x, y-40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, label2, (x, y-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(display_frame, label3, (x, y-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            except Exception as e:
                print(f"Error drawing rectangle: {e}")
        
        return display_frame
    
    def update_video_display(self, frame):
        """Update video display"""
        try:
            # Resize for display
            height, width = frame.shape[:2]
            max_width, max_height = 500, 400  # Slightly smaller for new layout
            
            if width > max_width or height > max_height:
                scale = min(max_width / width, max_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            tk_image = ImageTk.PhotoImage(pil_image)
            
            # Update label
            self.video_label.config(image=tk_image, text="")
            self.video_label.image = tk_image  # Keep reference
            
        except Exception as e:
            print(f"Video display error: {e}")
    
    def analysis_loop(self):
        """Face analysis loop with improved gender detection"""
        while self.is_running:
            try:
                if self.current_frame is not None and not self.analysis_active:
                    self.analysis_active = True
                    frame = self.current_frame.copy()
                    
                    # Use larger frame for better detection
                    analysis_frame = cv2.resize(frame, (640, 480))
                    
                    self.analyze_frame(analysis_frame)
                    self.analysis_active = False
                
                time.sleep(1.5)  # Analyze every 1.5 seconds for faster updates
                
            except Exception as e:
                print(f"Analysis loop error: {e}")
                self.analysis_active = False
                time.sleep(2)
    
    def analyze_frame(self, frame):
        """Analyze frame with enhanced emotion detection and adjusted age estimation"""
        try:
            print("Starting enhanced frame analysis...")
            
            # Try multiple approaches for better detection
            all_results = []
            
            # Method 1: Standard analysis with multiple backends
            backends = ['opencv', 'mtcnn', 'retinaface']
            
            for backend in backends:
                try:
                    print(f"Trying {backend} detector...")
                    results = DeepFace.analyze(frame, 
                                             actions=['emotion', 'age'],
                                             enforce_detection=False,
                                             silent=True,
                                             detector_backend=backend)
                    
                    if results:
                        all_results.append(('standard', backend, results))
                        print(f"Success with {backend}")
                        break  # Use first successful result
                        
                except Exception as e:
                    print(f"Failed with {backend}: {e}")
                    continue
            
            # Use the best available results
            if not all_results:
                print("All detection methods failed")
                return
            
            # Use standard results (first successful)
            method, backend, results = all_results[0]
            
            if not isinstance(results, list):
                results = [results]
            
            detected_faces = []
            
            for i, result in enumerate(results):
                try:
                    # Extract data
                    region = result.get('region', {})
                    emotions = result.get('emotion', {})
                    
                    # Enhanced emotion processing
                    if emotions:
                        enhanced_emotions = self.enhance_emotion_detection(emotions)
                        dominant_emotion = max(enhanced_emotions.items(), key=lambda x: x[1])[0]
                        
                        # Calculate mood score
                        mood_score = self.calculate_mood_score(enhanced_emotions)
                    else:
                        enhanced_emotions = emotions
                        dominant_emotion = result.get('dominant_emotion', 'neutral')
                        mood_score = 0
                    
                    # Adjust age estimation
                    detected_age = result.get('age', 25)
                    adjusted_age = max(0, int(detected_age * 0.85))  # Reduce age by 15%
                    
                    # Try to identify person
                    identity = self.identify_person_from_region(frame, region)
                    
                    face_data = {
                        'region': region,
                        'identity': identity,
                        'emotions': enhanced_emotions,
                        'dominant_emotion': dominant_emotion,
                        'mood_score': mood_score,
                        'age': adjusted_age,
                        'timestamp': time.time()
                    }
                    
                    detected_faces.append(face_data)
                    
                    # Store emotion and mood history
                    person_key = identity if identity != 'Unknown' else f'Unknown_{i}'
                    self.emotion_history[person_key].append((time.time(), enhanced_emotions))
                    self.mood_history[person_key].append((time.time(), mood_score))
                    
                    print(f"Final Detection: {identity}, {dominant_emotion} ({enhanced_emotions.get(dominant_emotion, 0):.1f}%), Mood: {mood_score:+.0f}, Age {adjusted_age}")
                    
                except Exception as e:
                    print(f"Error processing face {i}: {e}")
            
            self.detected_faces = detected_faces
            
            # Update GUI
            self.root.after(0, self.update_info_display)
            self.root.after(0, self.update_mood_graph)
            self.root.after(0, self.update_valence_arousal_plot)  # Add valence-arousal update
            
        except Exception as e:
            print(f"Analysis error: {e}")
            import traceback
            traceback.print_exc()
    
    def enhance_emotion_detection(self, emotions):
        """Enhance emotion detection for better sensitivity and accuracy"""
        if not emotions:
            return emotions
        
        enhanced = emotions.copy()
        
        # Boost sensitivity for subtle emotions
        if 'sad' in enhanced:
            if 10 < enhanced['sad'] < 30:  # Subtle sadness
                if enhanced.get('fear', 0) > 15 or enhanced.get('disgust', 0) > 10:
                    enhanced['sad'] = min(enhanced['sad'] * 1.3, 60)
                    print(f"Boosted 'sad' emotion: {enhanced['sad']:.1f}%")
        
        if 'angry' in enhanced:
            if 8 < enhanced['angry'] < 25:  # Subtle anger
                if enhanced.get('disgust', 0) > 10 or enhanced.get('fear', 0) > 10:
                    enhanced['angry'] = min(enhanced['angry'] * 1.4, 65)
                    print(f"Boosted 'angry' emotion: {enhanced['angry']:.1f}%")
        
        # Reduce dominance of 'neutral' if other emotions are present
        negative_emotions = (
            enhanced.get('sad', 0) +
            enhanced.get('angry', 0) +
            enhanced.get('fear', 0) +
            enhanced.get('disgust', 0)
        )
        if negative_emotions > 30 and enhanced.get('neutral', 0) > 50:
            enhanced['neutral'] = max(enhanced['neutral'] * 0.8, 20)
            print(f"Reduced 'neutral' emotion: {enhanced['neutral']:.1f}%")
        
        # Add custom rules for emotion enhancement
        if 'fear' in enhanced and 'disgust' in enhanced:
            if enhanced['fear'] > 15 and enhanced['disgust'] > 10:
                enhanced['sad'] = min(enhanced.get('sad', 0) + 10, 50)
                print(f"Boosted 'sad' due to 'fear' and 'disgust': {enhanced['sad']:.1f}%")
        
        # Normalize to ensure total adds up to ~100%
        total = sum(enhanced.values())
        if total > 0:
            enhanced = {k: (v / total) * 100 for k, v in enhanced.items()}
        
        return enhanced
    
    def identify_person_from_region(self, frame, region):
        """Identify person using local database"""
        try:
            if not self.local_face_data:
                return 'Unknown'
            
            x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
            
            if w <= 0 or h <= 0:
                return 'Unknown'
            
            # Extract face
            face = frame[y:y+h, x:x+w]
            if face.size == 0:
                return 'Unknown'
            
            # Try to match with local database
            for person_name, reference_path in self.local_face_data.items():
                try:
                    result = DeepFace.verify(face, reference_path, 
                                           enforce_detection=False, 
                                           silent=True,
                                           distance_metric='cosine')
                    
                    if result.get('verified', False) or result.get('distance', 1.0) < 0.4:
                        print(f"Matched: {person_name} (distance: {result.get('distance', 'N/A')})")
                        return person_name
                        
                except Exception as e:
                    print(f"Error matching against {person_name}: {e}")
                    continue
            
            return 'Unknown'
            
        except Exception as e:
            print(f"Identification error: {e}")
            return 'Unknown'
    
    def update_info_display(self):
        """Update information display with mood scores"""
        try:
            self.info_text.delete(1.0, tk.END)
            
            # Show database info
            self.info_text.insert(tk.END, f"📁 Local Database: {len(self.local_face_data)} people loaded\n")
            if self.local_face_data:
                names = ', '.join(list(self.local_face_data.keys())[:5])
                if len(self.local_face_data) > 5:
                    names += f" +{len(self.local_face_data)-5} more"
                self.info_text.insert(tk.END, f"   Known: {names}\n\n")
            else:
                self.info_text.insert(tk.END, "   Add images to local_images folder and refresh!\n\n")
            
            # Show detected faces
            if not self.detected_faces:
                self.info_text.insert(tk.END, "👤 No faces currently detected\n")
            else:
                for i, face_data in enumerate(self.detected_faces):
                    identity = face_data.get('identity', 'Unknown')
                    emotion = face_data.get('dominant_emotion', 'neutral')
                    mood_score = face_data.get('mood_score', 0)
                    age = face_data.get('age', 'N/A')
                    
                    # Mood description
                    if mood_score > 50:
                        mood_desc = "😄 Very Happy"
                    elif mood_score > 20:
                        mood_desc = "😊 Happy"
                    elif mood_score > -20:
                        mood_desc = "😐 Neutral"
                    elif mood_score > -50:
                        mood_desc = "😔 Sad"
                    else:
                        mood_desc = "😢 Very Sad"
                    
                    self.info_text.insert(tk.END, f"👤 Person {i+1}: {identity}\n")
                    self.info_text.insert(tk.END, f"   😊 Emotion: {emotion}\n")
                    self.info_text.insert(tk.END, f"   📊 Mood Score: {mood_score:+.0f} ({mood_desc})\n")
                    self.info_text.insert(tk.END, f"   📅 Age: {age}\n")
                    
                    # Top emotions
                    emotions = face_data.get('emotions', {})
                    if emotions:
                        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
                        self.info_text.insert(tk.END, "   📈 Top Emotions:\n")
                        for emo, score in sorted_emotions[:3]:
                            self.info_text.insert(tk.END, f"      {emo}: {score:.1f}%\n")
                    
                    self.info_text.insert(tk.END, "\n")
            
        except Exception as e:
            print(f"Info display error: {e}")
    
    def update_mood_graph(self):
        """Update mood score graph (Best vs Worst mood)"""
        try:
            self.mood_ax.clear()
            
            if not self.mood_history:
                self.mood_ax.text(0.5, 0.5, 'No mood data yet\nWait for face detection...', 
                            ha='center', va='center', transform=self.mood_ax.transAxes, fontsize=12)
                self.mood_canvas.draw()
                return
            
            self.mood_ax.set_title("Real-Time Mood Score Tracking")
            self.mood_ax.set_xlabel("Time (seconds ago)")
            self.mood_ax.set_ylabel("Mood Score (Best +100 ↔ Worst -100)")
            
            # Add horizontal reference lines
            self.mood_ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Neutral')
            self.mood_ax.axhline(y=50, color='green', linestyle=':', alpha=0.5, label='Happy')
            self.mood_ax.axhline(y=-50, color='red', linestyle=':', alpha=0.5, label='Sad')
            
            current_time = time.time()
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
            
            plotted = False
            for i, (person_id, mood_data) in enumerate(self.mood_history.items()):
                if len(mood_data) < 2:
                    continue
                
                # Get recent data (last 60 seconds)
                recent_data = [(t, mood_score) for t, mood_score in mood_data if current_time - t <= 60]
                
                if len(recent_data) < 2:
                    continue
                
                times = [current_time - t for t, _ in recent_data]
                mood_scores = [mood_score for _, mood_score in recent_data]
                
                color = colors[i % len(colors)]
                self.mood_ax.plot(times, mood_scores, 
                           marker='o', label=person_id, color=color, linewidth=2, markersize=6)
                plotted = True
            
            if plotted:
                self.mood_ax.legend()
                self.mood_ax.grid(True, alpha=0.3)
                self.mood_ax.set_xlim(60, 0)
                self.mood_ax.set_ylim(-100, 100)
                
                # Color the background regions
                self.mood_ax.axhspan(50, 100, alpha=0.1, color='green', label='Happy Zone')
                self.mood_ax.axhspan(-50, -100, alpha=0.1, color='red', label='Sad Zone')
                self.mood_ax.axhspan(-50, 50, alpha=0.05, color='gray', label='Neutral Zone')
            else:
                self.mood_ax.text(0.5, 0.5, 'Collecting mood data...', 
                            ha='center', va='center', transform=self.mood_ax.transAxes, fontsize=12)
            
            self.mood_canvas.draw()
            
        except Exception as e:
            print(f"Graph update error: {e}")
    
    def refresh_database(self):
        """Refresh local images database"""
        try:
            self.load_local_faces()
            self.update_info_display()
            messagebox.showinfo("Success", 
                f"Database refreshed!\nLoaded {len(self.local_face_data)} people.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh database: {e}")
    
    def open_images_folder(self):
        """Open local images folder"""
        try:
            abs_path = os.path.abspath(self.local_images_db)
            if platform.system() == "Windows":
                subprocess.run(['explorer', abs_path], check=True)
            elif platform.system() == "Darwin":
                subprocess.run(['open', abs_path], check=True)
            else:
                subprocess.run(['xdg-open', abs_path], check=True)
        except Exception as e:
            messagebox.showinfo("Folder Location", 
                f"Could not open folder automatically.\n\nPath: {os.path.abspath(self.local_images_db)}")
    
    def create_face_selection_dialog(self, face_options):
        """Create a simple dialog for selecting a face."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Face")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="Select a face to save:").pack(pady=10)

        listbox = tk.Listbox(dialog, selectmode=tk.SINGLE)
        for option in face_options:
            listbox.insert(tk.END, option)
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        result = {"selection": None}

        def on_select():
            selected_index = listbox.curselection()
            if selected_index:
                result["selection"] = selected_index[0]
            dialog.destroy()

        tk.Button(dialog, text="OK", command=on_select).pack(pady=10)

        self.root.wait_window(dialog)
        return result["selection"]

    def save_new_face(self):
        """Save current face as a new person in the database"""
        if not self.is_running or self.current_frame is None:
            messagebox.showwarning("Warning", "Please start the camera first!")
            return
        
        if not self.detected_faces:
            messagebox.showwarning("Warning", "No faces detected in current frame!")
            return
        
        # Get person name from user
        name = simpledialog.askstring(
            "Save New Face", 
            "Enter the person's name:",
            initialvalue=""
        )
        
        if not name or not name.strip():
            return
        
        name = name.strip()
        
        try:
            # If multiple faces detected, let user choose which one to save
            if len(self.detected_faces) > 1:
                face_options = []
                for i, face_data in enumerate(self.detected_faces):
                    identity = face_data.get('identity', f'Unknown_{i+1}')
                    emotion = face_data.get('dominant_emotion', 'neutral')
                    age = face_data.get('age', 'N/A')
                    face_options.append(f"Face {i+1}: {identity} ({emotion}, Age: {age})")
                
                selected_face_index = self.create_face_selection_dialog(face_options)
                
                if selected_face_index is None:
                    return
            else:
                selected_face_index = 0
            
            # Get the selected face region
            face_data = self.detected_faces[selected_face_index]
            region = face_data.get('region', {})
            
            x = int(region.get('x', 0))
            y = int(region.get('y', 0))
            w = int(region.get('w', 0))
            h = int(region.get('h', 0))
            
            if w <= 0 or h <= 0:
                messagebox.showerror("Error", "Invalid face region detected!")
                return
            
            # Extract face from current frame with some padding
            padding = 20
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(self.current_frame.shape[1], x + w + padding)
            y_end = min(self.current_frame.shape[0], y + h + padding)
            
            face_image = self.current_frame[y_start:y_end, x_start:x_end]
            
            if face_image.size == 0:
                messagebox.showerror("Error", "Could not extract face image!")
                return
            
            # Create filename (replace spaces with underscores)
            safe_name = name.replace(' ', '_').lower()
            # Remove any special characters
            safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')
            
            filename = f"{safe_name}.jpg"
            filepath = os.path.join(self.local_images_db, filename)
            
            # Save the face image
            success = cv2.imwrite(filepath, face_image)
            
            if success:
                # Refresh the database
                self.load_local_faces()
                self.update_info_display()
                
                messagebox.showinfo("Success", 
                    f"Face saved successfully!\n\nName: {name}\nFile: {filename}\n\nDatabase refreshed automatically.")
                
                print(f"✅ Saved new face: {name} -> {filename}")
            else:
                messagebox.showerror("Error", "Failed to save face image!")
                
        except Exception as e:
            print(f"Error saving face: {e}")
            messagebox.showerror("Error", f"Failed to save face: {str(e)}")
    
    def run(self):
        """Run the application"""
        def on_closing():
            self.stop_system()
            self.root.destroy()
        
        self.root.protocol("WM_DELETE_WINDOW", on_closing)
        self.root.mainloop()


if __name__ == "__main__":
    """
    Enhanced Emotion & Identity Recognition System with Valence-Arousal Graph
    
    SETUP INSTRUCTIONS:
    1. Install requirements: pip install deepface opencv-python matplotlib pillow
    2. Run this script
    3. Add photos to the 'local_images' folder (click 'Open Images Folder')
    4. Name images like: john_doe.jpg, jane_smith.png
    5. Click 'Refresh Database' 
    6. Click 'Start Camera'
    
    NEW FEATURES IN THIS VERSION:
    - Added Valence-Arousal Graph showing emotion positions in 2D space
    - Real-time emotion plotting with person tracking
    - Interactive graph with quadrant labels and reference emotions
    - Enhanced 3-column layout for better space management
    - Proper error handling for graph operations
    
    The Valence-Arousal graph shows:
    - Valence: How positive (right) or negative (left) an emotion is
    - Arousal: How energetic (up) or calm (down) an emotion is
    - Current detected emotions as colored dots with person labels
    - Reference emotion positions for comparison
    """
    
    print("="*60)
    print("🎯 ENHANCED EMOTION & IDENTITY RECOGNITION SYSTEM")
    print("="*60)
    print("✅ Added Valence-Arousal Graph for emotion positioning")
    print("✅ Real-time emotion tracking in 2D space")
    print("✅ Enhanced 3-column layout with proper space management")
    print("✅ Comprehensive error handling for all graph operations")
    print("="*60)
    
    try:
        app = EmotionIdentityRecognizer()
        if hasattr(app, 'root'):  # Check if initialization was successful
            app.run()
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        input("Press Enter to exit...")
