import os
import cv2
import numpy as np
from yt_dlp import YoutubeDL
import yaml

FRAME_PER_VIDEO = 100

# Create directories if they don't exist
os.makedirs('videos', exist_ok=True)
os.makedirs('frames', exist_ok=True)

def download_video(url, output_path='videos'):
    ydl_opts = {
        'format': 'best[height<=480]',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'noplaylist': True
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        video_info = ydl.extract_info(url, download=False)
        video_filename = ydl.prepare_filename(video_info)
    return video_filename

def extract_frames(video_path, output_path='frames', frame_count=FRAME_PER_VIDEO):
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = length // frame_count
    
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    
    for i in range(frame_count):
        frame_no = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Resize the frame while maintaining aspect ratio
        frame = resize_frame(frame, (224, 224))
        frame_filename = os.path.join(output_path, f"{os.path.basename(video_path)}_frame_{i}.jpg")
        cv2.imwrite(frame_filename, frame)
    
    cap.release()

def resize_frame(frame, target_size):
    h, w = frame.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    frame_resized = cv2.resize(frame, (nw, nh))
    top_pad = (target_size[0] - nh) // 2
    bottom_pad = target_size[0] - nh - top_pad
    left_pad = (target_size[1] - nw) // 2
    right_pad = target_size[1] - nw - left_pad
    frame_padded = cv2.copyMakeBorder(frame_resized, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return frame_padded

# Read YouTube URLs from a YAML file
def read_urls_from_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data['games']

# Read cached URLs from file
def read_cached_urls(file_path='cached_urls.txt'):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return set(line.strip() for line in file)
    return set()

# Write URL to cache file
def write_url_to_cache(url, file_path='cached_urls.txt'):
    with open(file_path, 'a') as file:
        file.write(url + '\n')

# Specify the path to the YAML file containing the YouTube URLs
url_file_path = 'youtube_urls.yaml'
games_urls = read_urls_from_yaml(url_file_path)
cached_urls = read_cached_urls()

for game, urls in games_urls.items():
    game_frame_path = os.path.join('frames', game)
    os.makedirs(game_frame_path, exist_ok=True)
    
    for url in urls:
        if url in cached_urls:
            print(f"Skipping already downloaded video: {url}")
            continue
        
        try:
            print(f"Downloading video from {url}")
            video_path = download_video(url)
            print(f"Extracting frames from {video_path}")
            extract_frames(video_path, output_path=game_frame_path)
            write_url_to_cache(url)
        except Exception as e:
            print(f"Failed to process {url}: {e}")