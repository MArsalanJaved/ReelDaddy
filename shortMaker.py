from pytube import YouTube
import cv2
import pandas as pd
import subprocess
from openai import OpenAI
import numpy as np
import json
import math
import pdb
import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip
import json
from tqdm.notebook import tqdm
import openai
import logging
import json
from youtube_transcript_api import YouTubeTranscriptApi
import sys
import yt_dlp
import textwrap

video_folder = 'videos/'
output_folder = 'reels/'

api_key = ''  # Replace with your actual OpenAI API key
client = OpenAI(
    api_key=api_key
)

net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

zoom_factor = 2.0  # Adjust this value to zoom in (e.g., 1.2) or out (e.g., 0.8)
face_change_threshold = 400  


os.makedirs(output_folder, exist_ok=True)

def get_video_title(yt_id):
    url = f'https://www.youtube.com/watch?v={yt_id}'
    
    def list_formats(url):
        ydl_opts = {
            'listformats': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info
    
    info = list_formats(url)
    
    video_title = info.get('title', 'video').replace('/', '_').replace('\\', '_').replace(' ', '_')
    
    return video_title
    
def convert_transcript_to_subtitles(transcript):
    subtitles = []
    for entry in transcript:
        subtitle = {
            "text": entry["text"],
            "start": entry["start"],
            "end": entry["start"] + entry["duration"]
        }
        subtitles.append(subtitle)
    return subtitles

def segment_video(response, input_file, video_id):
    try:
        output_folder = "videos"
        
        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        for i, segment in enumerate(response):
            start_time = math.floor(float(segment.get("start_time", 0)))
            end_time = math.ceil(float(segment.get("end_time", 0))) + 2
            output_file = os.path.join(output_folder, f"{video_id}_{str(i).zfill(3)}.mp4")
            
            with VideoFileClip(input_file) as video:
                subclip = video.subclip(start_time, end_time)
                subclip.write_videofile(output_file, codec="libx264", audio_codec="aac")
                
            segment["output_file"] = output_file

            # csv_file_path = 'parsed_content.csv'
            # df = pd.read_csv(csv_file_path)
            # new_row = pd.DataFrame([{
            #     'start_time': segment['start_time'],
            #     'end_time': segment['end_time'],
            #     'description': segment['description'],
            #     'duration': segment['duration'],
            #     'output_file': segment['output_file'],
            #     'reel': '',
            #     'posted_twitter': False,
            #     'posted_instagram': False
            # }])
            # df = pd.concat([df, new_row], ignore_index=True)
            # df.to_csv(csv_file_path, index=False)
            
    except Exception as e:
        print(e)
        pass

def get_transcript(video_id):
    # Get the transcript for the given YouTube video ID
    transcript = YouTubeTranscriptApi.get_transcript(video_id)

    # Format the transcript for feeding into GPT-4
    formatted_transcript = ''
    for entry in transcript:
        start_time = "{:.2f}".format(entry['start'])
        end_time = "{:.2f}".format(entry['start'] + entry['duration'])
        text = entry['text']
        formatted_transcript += f"{start_time} --> {end_time} : {text}\n"

    return transcript


def distance(box1, box2):
    # Calculate Euclidean distance between two boxes
    (x1, y1, x2, y2) = box1
    (a1, b1, a2, b2) = box2
    return np.sqrt((x1 - a1)**2 + (y1 - b1)**2 + (x2 - a2)**2 + (y2 - b2)**2)

def get_subtitles_for_segment(output_file, subtitle_objects):
    for obj in subtitle_objects:
        try:
            if obj['output_file'] == output_file:
                return obj['subtitles']
        except Exception as e:
            print(e)
            pass
    return []
    
def analyze_transcript(transcripts, video_id, chunk_size=5000, overlap=50):
    response_obj = '''[
    {
        "start_time": 97.19,
        "end_time": 127.43,
        "description": "One liner of the section with hashtags which is tweet long. This will be displade on the video header",
        "duration": 36
    },
    {
        "start_time": 169.58,
        "end_time": 199.10,
        "description": "One liner of the section with hashtags which is tweet long. This will be displade on the video header",
        "duration": 33
    }
    ]'''

    # Combine all the texts in the transcripts into one big string
    all_text = ' '.join([segment['text'] for segment in transcripts])

    # Split the text into words and make chunks of words
    words = all_text.split(' ')
    chunks = []
    index = 0
    title = get_video_title(video_id)
    while index < len(words):
        start = max(0, index - overlap)
        chunk = " ".join(words[start: index + chunk_size])
        chunks.append(chunk)
        index += chunk_size

    logging.info(f'Transcript has been chunked into {len(chunks)} chunks')

    # Process each chunk separately and collect responses
    responses = []
    for idx, chunk in enumerate(chunks, start=1):
        logging.info(f'Processing chunk {idx}/{len(chunks)}')

        prompt = f"""This is a transcript of a video/podcast. 
        Please identify the most viral sections from this part of the video, 
        make sure they are more than 120 seconds in duration and not more than 250 seconds, 
        and provide extremely accurate timestamps. 
        Ensure you capture the entire question and answer for each section.
        Respond only in this format: {response_obj}. 
        Make sure that the return format is JSON and syntactically correct.
        I just want JSON as Response (nothing else).
        \n\nHere is the Transcription:\n{chunk}"""

        
        messages = [
            {"role": "system", "content": "You are a ViralGPT helpful assistant. You are master at reading YouTube transcripts and identifying the most interesting parts and viral content from the podcasts."},
            {"role": "user", "content": prompt}
        ]

        logging.info(f'Sending chunk {idx}/{len(chunks)} to the model')
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=512,
            n=1,
            stop=None
        )
        logging.info(f'Received response for chunk {idx}/{len(chunks)}')
        response_content = response.choices[0].message.content
        responses.append(response_content)
        logging.info(f'Added response for chunk {idx} to responses')
        
    # Combine all responses into a single response
    combined_response = ' '.join(responses)
    logging.info('Completed processing all chunks')

    return responses

def convert_transcript_to_subtitles(transcript):
    subtitles = []
    for entry in transcript:
        subtitle = {
            "text": entry["text"],
            "start": entry["start"],
            "end": entry["start"] + entry["duration"]
        }
        subtitles.append(subtitle)
    return subtitles

def create_subtitles_for_segments(transcript, parsed_content):
    subtitle_objects = []
    
    for segment in parsed_content:
        try:
            subtitles = []
            for entry in transcript:
                if segment['start_time'] <= entry['start'] < segment['end_time']:
                    subtitle = {
                        "text": entry['text'],
                        "start": entry['start'] - segment['start_time'],
                        "end": entry['start'] + entry['duration'] - segment['start_time']
                    }
                    subtitles.append(subtitle)
            
            subtitle_object = {
                'output_file': segment['output_file'],
                'subtitles': subtitles
            }
            subtitle_objects.append(subtitle_object)
        except Exception as e:
            print(e)
            pass
    
    return subtitle_objects


import re

def remove_hashtags(text):
    # Regular expression to find hashtags
    return re.sub(r'#\S+', '', text).strip()

def get_shorts(video_id, parsed_content, width=1080, height=1920, logo_path='logo.png'):
    try:
        video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4") and video_id in f]
        for one_parsed_content in tqdm(parsed_content, desc="Processing videos"): 
            file_name = one_parsed_content['output_file'].split('/')[-1]
            input_video_path = os.path.join(video_folder, file_name)
            temp_video_path = os.path.join(output_folder, f'{video_id}_temp_' + file_name)
            output_video_path = os.path.join(output_folder, file_name)

            # Load the video
            cap = cv2.VideoCapture(input_video_path)
            original_frame_rate = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, original_frame_rate, (width, height))

            crop_box = None  # To store the current crop region
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            for _ in tqdm(range(frame_count), desc=f"Processing frames for {file_name}"):
                ret, frame = cap.read()
                if not ret:
                    break

                (h, w) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                net.setInput(blob)
                detections = net.forward()

                detected_face = None
                for i in range(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:  # confidence threshold
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        # Adjust the bounding box based on zoom factor
                        box_w = (endX - startX) * zoom_factor
                        box_h = (endY - startY) * zoom_factor
                        center_x = startX + (endX - startX) // 2
                        center_y = startY + (endY - startY) // 2

                        startX = int(center_x - box_w // 2)
                        startY = int(center_y - box_h // 2)
                        endX = int(center_x + box_w // 2)
                        endY = int(center_y + box_h // 2)

                        # Ensure the coordinates are within the frame bounds
                        startX = max(0, startX)
                        startY = max(0, startY)
                        endX = min(w, endX)
                        endY = min(h, endY)

                        detected_face = (startX, startY, endX, endY)
                        break

                if detected_face:
                    if crop_box is None or distance(detected_face, crop_box) > face_change_threshold:
                        crop_box = detected_face

                if crop_box:
                    (startX, startY, endX, endY) = crop_box
                    # Crop the frame around the current region
                    cropped_frame = frame[startY:endY, startX:endX]

                    # Calculate the scaling factor to fit the frame
                    scale_w = width / (endX - startX)
                    scale_h = height / (endY - startY)
                    scale = min(scale_w, scale_h)

                    new_w = int((endX - startX) * scale)
                    new_h = int((endY - startY) * scale)
                    resized_frame = cv2.resize(cropped_frame, (new_w, new_h))
                else:
                    # If no person detected, resize the entire frame while maintaining the aspect ratio
                    scale_w = width / w
                    scale_h = height / h
                    scale = min(scale_w, scale_h)

                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    resized_frame = cv2.resize(frame, (new_w, new_h))

                # Create a blank frame with the desired dimensions and center the resized frame
                blank_frame = np.zeros((height, width, 3), dtype=np.uint8)
                x_offset = (width - resized_frame.shape[1]) // 2
                y_offset = (height - resized_frame.shape[0]) // 2
                blank_frame[y_offset:y_offset + resized_frame.shape[0], x_offset:x_offset + resized_frame.shape[1]] = resized_frame
                final_frame = blank_frame

                out.write(final_frame)

            cap.release()
            out.release()

            # Load the original video and extract the audio
            original_clip = VideoFileClip(input_video_path)
            audio_clip = original_clip.audio

            # Load the processed video without audio
            video_clip = VideoFileClip(temp_video_path)

            # Combine the video with the extracted audio
            final_clip = video_clip.set_audio(audio_clip)
            final_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac', bitrate='5000k', fps=original_frame_rate)
            one_parsed_content['reel'] = output_video_path
            # Remove the temporary video file
            os.remove(temp_video_path)

    except Exception as e:
        print(f"Exception: {e}")

def distance(box1, box2):
    # Calculate the Euclidean distance between two boxes (x, y, width, height)
    center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    return ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python download_video.py <youtube_video_id>")
        sys.exit(1)

    video_id = sys.argv[1]
    transcript = get_transcript(video_id)
    
    interesting_segment = analyze_transcript(transcript, video_id)
    
    parsed_content = []
    for seg in interesting_segment:
        seg = seg.replace('```','').replace('json','')
        try:
            parsed_content += json.loads(seg)
        except json.JSONDecodeError as e:
            print(f"Exception: {e}")
    
    segment_video(parsed_content, f'initialVideos/{video_id}.mp4', video_id)
    
    all_subtitles = create_subtitles_for_segments(transcript, parsed_content)
    face_change_threshold = 400  
    zoom_factor = 5
    get_shorts(video_id, parsed_content, width=720, height=1080)


