import os
import cv2
import numpy as np
import argparse
import yaml
from yt_dlp import YoutubeDL

def download_video(url, output_path='videos', ignore_cache=False):
    ydl_opts = {
        'format': 'best[height<=720]',
        'outtmpl': os.path.join(output_path, '%(id)s.%(ext)s'),
        'noplaylist': True
    }
    if ignore_cache or not os.path.exists(os.path.join(output_path, f"{get_video_id(url)}.mp4")):
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            video_info = ydl.extract_info(url, download=False)
            video_filename = ydl.prepare_filename(video_info)
        return video_filename, video_info['id']
    else:
        return os.path.join(output_path, f"{get_video_id(url)}.mp4"), get_video_id(url)

def get_video_id(url):
    with YoutubeDL({'quiet': True}) as ydl:
        video_info = ydl.extract_info(url, download=False)
        return video_info['id']

def extract_frames(video_path, video_id, output_path='frames', interval_sec=60, max_size=570):
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
        frame = resize_frame(frame, max_size)
        frame_filename = os.path.join(output_path, f"{video_id}_{frame_index}.jpg")
        cv2.imwrite(frame_filename, frame)
        
        frame_no += frame_interval
        frame_index += 1
    
    cap.release()

def resize_frame(frame, max_size):
    h, w = frame.shape[:2]
    if h > w:
        scale = max_size / h
    else:
        scale = max_size / w
    nh, nw = int(h * scale), int(w * scale)
    frame_resized = cv2.resize(frame, (nw, nh))
    return frame_resized

def read_urls_from_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data['games']

def read_cached_urls(file_path='cached_urls.txt'):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return set(line.strip() for line in file)
    return set()

def write_url_to_cache(url, file_path='cached_urls.txt'):
    with open(file_path, 'a') as file:
        file.write(url + '\n')

def clean_output_folders():
    folders = ['videos', 'frames', 'cached_urls.txt']
    for folder in folders:
        if os.path.isdir(folder):
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    try:
                        os.rmdir(file_path)
                    except OSError:
                        clean_directory(file_path)
                        os.rmdir(file_path)
        elif os.path.isfile(folder):
            os.remove(folder)

def clean_directory(directory):
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            clean_directory(file_path)
            os.rmdir(file_path)

def clean_frames_folder():
    frames_folder = 'frames'
    if os.path.isdir(frames_folder):
        for game_folder in os.listdir(frames_folder):
            game_folder_path = os.path.join(frames_folder, game_folder)
            if os.path.isdir(game_folder_path):
                for file in os.listdir(game_folder_path):
                    file_path = os.path.join(game_folder_path, file)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)

def main():
    parser = argparse.ArgumentParser(description='Download videos and extract frames.')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the input YAML file.')
    parser.add_argument('--interval_sec', type=int, default=60, help='Interval in seconds between frames.')
    parser.add_argument('--max_size', type=int, default=570, help='Maximum size of the frame dimension.')
    parser.add_argument('--test', action='store_true', help='Test run that processes only 1 video from each game.')
    parser.add_argument('--ignore_video_cache', action='store_true', help='Ignore video cache and redownload videos.')
    parser.add_argument('--clean', action='store_true', help='Clean the output folders before starting.')
    parser.add_argument('--clean-frames', action='store_true', help='Clean the frames folder before starting.')
    parser.add_argument('--skip-frames', action='store_true', help='Skip frames generation when video is cached.')


    args = parser.parse_args()

    if args.clean:
        clean_output_folders()

    if args.clean_frames:
        clean_frames_folder()

    games_urls = read_urls_from_yaml(args.input)
    cached_urls = read_cached_urls()

    for game, urls in games_urls.items():
        game_frame_path = os.path.join('frames', game)
        os.makedirs(game_frame_path, exist_ok=True)
        
        if args.test:
            urls = urls[:1]

        for url in urls:
            video_filename = None
            video_id = None
            
            if url in cached_urls and not args.ignore_video_cache:
                print(f"Skipping download for cached video: {url}")
                video_filename = os.path.join('videos', f"{get_video_id(url)}.mp4")
                video_id = get_video_id(url)
                if args.skip_frames:
                        print(f"Skipping frame extraction for cached video: {url}")
                        continue
            else:
                try:
                    print(f"Downloading video from {url}")
                    video_filename, video_id = download_video(url, ignore_cache=args.ignore_video_cache)
                    write_url_to_cache(url)
                except Exception as e:
                    print(f"Failed to process {url}: {e}")
                    continue
            
            if video_filename and video_id:
                print(f"Extracting frames from {video_filename}")
                extract_frames(video_filename, video_id, output_path=game_frame_path, interval_sec=args.interval_sec, max_size=args.max_size)

if __name__ == '__main__':
    main()
