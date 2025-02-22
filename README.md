# Live Member Face Detection
## Project Description
This project was inspired by my experience waiting in long lines to have my season pass scanned at a ski resort. I realized there had to be a more efficient way to verify memberships and grant access to members without unnecessary delays. The same issue exists in places like Costco, sporting events, and other venues where long queues form just to validate entry credentials.
![Snow Example](images/snow_ex.jpg)

## Project Proposal
To solve this problem, I developed a real-time face recognition system using computer vision. <br>
The goal is to eliminate long wait times by allowing members to gain access seamlessly through facial recognition, reducing the need for manual verification. <br>

![john Example](images/john_scan.jpg)

### System Overview
This system leverages sqlite to develop a members database, OpenCV, Ultralytics' for YOLO for object detection, and LBPHFaceRecognizer for real-time face recognition.<br>
By implementing this solution, venues such as ski resorts, retail stores, and stadiums can enhance customer experience by providing frictionless, secure, and efficient entry for their members.
### Table of Contents  
- [Setting up the SQLite Database](#setting-up-the-sqlite-database)  
- [Acquiring Video & Splitting Videos into Still Images](#acquiring-video--splitting-videos-into-still-images)  
- [Building YOLO Model](#building-yolo-model)  
- [Building LBPHFaceRecognizer](#building-lbphfacerecognizer)  
- [Future Implementations](#future-implementations)  
### Setting up the SQLite Database
Before creating the SQLite database, I needed to gather volunteers for my LMFD project.<br>
After reaching out to friends and family, I enlisted 12 volunteers, each assigned a unique ID.
##### Member List  

| Name      | ID Number |
|-----------|----------|
| Angela    | 1        |
| Classmate | 2        |
| Giuliana  | 3        |
| Javier    | 4        |
| John      | 5        |
| Maite     | 6        |
| Mike      | 7        |
| Ron       | 8        |
| Shanti    | 9        |
| Tom       | 10       |
| Vilma     | 11       |
| Will      | 12       |
#### Storing Data
For each volunteer, I generated synthetic data to store 15 different demographic attributes. <br> This data was randomly created and represents the type of information that potential members might have.
The Members database consisted of the following:
### Members Database

| Column           | Column Type         |
|------------------|---------------------|
| id               | INTEGER PRIMARY KEY |
| name             | TEXT                |
| age              | INTEGER             |
| date_of_birth    | TEXT                |
| address          | TEXT                |
| loyalty          | INTEGER             |
| member_since     | TEXT                |
| gender           | TEXT                |
| email            | TEXT                |
| phone_number     | TEXT                |
| membership_type  | TEXT                |
| status           | TEXT                |
| occupation       | TEXT                |
| interest         | TEXT                |
| marital_status   | TEXT                |
### SQlite Database Format
![ds](images/ds_example.jpg)
### Acquiring Video & Splitting Videos into Still Images 
I had all 12 of my volunteers send me an approx 30 second video of their face.
I suggested that in the video take a wide variety of their face so the yolo and LBPHFaceRecognizer can lern asepects of their face
For instance the video showed their face looking at the camera 
[![Video Thumbnail](images/john_ex.jpg)](images/john.mp4)
