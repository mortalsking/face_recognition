import cv2
import os
import numpy as np
import pickle
from datetime import datetime

class SimpleFaceRecognition:
    def __init__(self, data_directory="face_data"):
        self.data_directory = data_directory
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_data = []
        self.face_labels = []
        self.label_names = {}
        self.label_counter = 0
        
       
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
            
       
        self.load_data()

    def detect_faces(self, frame):
        """Detect faces in the given frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return gray, faces

    def register_new_face(self, name):
        """Register a new face with the given name"""
        if name in self.label_names.values():
            print(f"Name '{name}' already registered.")
            return False
            
        cap = cv2.VideoCapture(0)
        face_samples = []
        count = 0
        
        print(f"Registering new face for {name}...")
        print("Please look at the camera. Taking 20 samples...")
        
        while count < 20:
            ret, frame = cap.read()
            if not ret:
                continue
                
            gray, faces = self.detect_faces(frame)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                if count < 20:
                    face_sample = gray[y:y+h, x:x+w]
                    face_sample = cv2.resize(face_sample, (100, 100))
                    face_samples.append(face_sample)
                    count += 1
                    
            cv2.putText(frame, f"Samples: {count}/20", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Face Registration", frame)
            
            
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
        if count == 20:
           
            new_label = self.label_counter
            self.label_counter += 1
            self.label_names[new_label] = name
            
            
            for face_sample in face_samples:
                self.face_data.append(face_sample)
                self.face_labels.append(new_label)
                
            print(f"Successfully registered {name}")
            self.train_model()
            self.save_data()
            return True
        else:
            print("Failed to collect enough samples")
            return False

    def train_model(self):
        """Train the face recognition model"""
        if len(self.face_data) == 0:
            print("No face data available for training")
            return False
            
        print("Training model...")
        self.recognizer.train(self.face_data, np.array(self.face_labels))
        print("Model training complete")
        return True

    def save_data(self):
        """Save the model and label data"""
       
        self.recognizer.write(os.path.join(self.data_directory, "face_model.xml"))
        
        
        with open(os.path.join(self.data_directory, "label_data.pkl"), "wb") as f:
            data = {
                "label_names": self.label_names,
                "label_counter": self.label_counter
            }
            pickle.dump(data, f)
            
      
        with open(os.path.join(self.data_directory, "training_data.pkl"), "wb") as f:
            data = {
                "face_data": self.face_data,
                "face_labels": self.face_labels
            }
            pickle.dump(data, f)
            
        print("Data saved successfully")

    def load_data(self):
        """Load saved model and label data"""
        model_path = os.path.join(self.data_directory, "face_model.xml")
        label_path = os.path.join(self.data_directory, "label_data.pkl")
        training_path = os.path.join(self.data_directory, "training_data.pkl")
        
        if os.path.exists(model_path) and os.path.exists(label_path) and os.path.exists(training_path):
           
            self.recognizer.read(model_path)
            
           
            with open(label_path, "rb") as f:
                data = pickle.load(f)
                self.label_names = data["label_names"]
                self.label_counter = data["label_counter"]
                
            
            with open(training_path, "rb") as f:
                data = pickle.load(f)
                self.face_data = data["face_data"]
                self.face_labels = data["face_labels"]
                
            print("Loaded existing model and label data")
            return True
        else:
            print("No existing data found")
            return False
    
    def start_recognition(self):
        """Start real-time face recognition"""
        if len(self.label_names) == 0:
            print("No faces registered. Please register at least one face first.")
            return
            
        cap = cv2.VideoCapture(0)
        attendance_log = {}
        
        print("Starting face recognition. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
                
            gray, faces = self.detect_faces(frame)
            
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (100, 100))
                
               
                try:
                    label, confidence = self.recognizer.predict(face_roi)
                    
                   
                    if confidence < 70:  
                        name = self.label_names.get(label, "Unknown")
                        
                        # Log attendance
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        if name not in attendance_log:
                            attendance_log[name] = current_time
                            print(f"Attendance marked for {name} at {current_time}")
                        
                        
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{name} ({confidence:.1f})", 
                                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, "Unknown", (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                except:
                   
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, "Error", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            
            cv2.imshow("Face Recognition", frame)
            
           
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
       
        log_path = os.path.join(self.data_directory, f"attendance_{datetime.now().strftime('%Y%m%d')}.txt")
        with open(log_path, "w") as f:
            f.write("Name,Time\n")
            for name, time in attendance_log.items():
                f.write(f"{name},{time}\n")
                
        print(f"Attendance log saved to {log_path}")



def main():
    print("===== Simple Face Recognition System =====")
    
    app = SimpleFaceRecognition()
    
    while True:
        print("\nOptions:")
        print("1. Register a new face")
        print("2. Start face recognition")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            name = input("Enter the person's name: ")
            app.register_new_face(name)
        elif choice == '2':
            app.start_recognition()
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()