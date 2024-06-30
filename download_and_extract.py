import os
import cv2
import numpy as np
from yt_dlp import YoutubeDL
import yaml

# Create directories if they don't exist
os.makedirs('videos', exist_ok=True)
os.makedirs('frames', exist_ok=True)

def download_video(url, output_path='videos'):
    ydl_opts = {
        'format': 'best[height<=480]',
        'outtmpl': os.path.join(output_path, '%(id)s.%(ext)s'),
        'noplaylist': True
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        video_info = ydl.extract_info(url, download=False)
        video_filename = ydl.prepare_filename(video_info)
    return video_filename, video_info['id']

def extract_frames(video_path, video_id, output_path='frames', interval_sec=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_sec)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    
    frame_no = 0
    frame_index = 0
    while frame_no < length:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize the frame while maintaining aspect ratio
        frame = resize_frame(frame, (224, 224))
        frame_filename = os.path.join(output_path, f"{video_id}_{frame_index}.jpg")
        cv2.imwrite(frame_filename, frame)
        
        frame_no += frame_interval
        frame_index += 1
    
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
        video_filename = None
        video_id = None
        
        if url in cached_urls:
            print(f"Skipping download for cached video: {url}")
            # Attempt to find the video file in the videos folder
            try:
                with YoutubeDL({'quiet': True}) as ydl:
                    video_info = ydl.extract_info(url, download=False)
                    video_id = video_info['id']
                    video_filename = os.path.join('videos', f"{video_id}.mp4")
                    if not os.path.exists(video_filename):
                        raise FileNotFoundError(f"Video file for cached URL {url} not found.")
            except Exception as e:
                print(f"Failed to retrieve info for cached video {url}: {e}")
                continue
        else:
            try:
                print(f"Downloading video from {url}")
                video_filename, video_id = download_video(url)
                write_url_to_cache(url)
            except Exception as e:
                print(f"Failed to process {url}: {e}")
                continue
        
        if video_filename and video_id:
            print(f"Extracting frames from {video_filename}")
            extract_frames(video_filename, video_id, output_path=game_frame_path, interval_sec=60)
