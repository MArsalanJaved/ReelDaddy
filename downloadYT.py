import yt_dlp
import os
import subprocess
import sys

def list_formats(url):
    ydl_opts = {
        'listformats': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info

def download_best_video_audio(yt_id, save_path='initialVideos'):
    url = f'https://www.youtube.com/watch?v={yt_id}'
    info = list_formats(url)
    
    # Ensure the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Find the best video and audio format IDs
    best_video = None
    best_audio = None
    
    for f in info['formats']:
        if 'vcodec' in f and f['vcodec'] != 'none' and (best_video is None or f['height'] > best_video.get('height', 0)):
            best_video = f
        if 'acodec' in f and f['acodec'] != 'none' and (best_audio is None or f.get('abr', 0) > best_audio.get('abr', 0)):
            best_audio = f
    
    if not best_video or not best_audio:
        print("No suitable video or audio streams found.")
        return

    video_id = best_video['format_id']
    audio_id = best_audio['format_id']
    video_title = info.get('title', 'video').replace('/', '_').replace('\\', '_').replace(' ', '_')
    channel_name = info.get('uploader', 'unknown').replace('/', '_').replace('\\', '_').replace(' ', '_')

    print(f"Best video ID: {video_id}, resolution: {best_video['height']}p")
    print(f"Best audio ID: {audio_id}, bitrate: {best_audio.get('abr', 'unknown')}kbps")
    print(f"Video title: {video_title}")
    print(f"Channel Name: {channel_name}")
    print(f"Video ID: {yt_id}")

    video_opts = {
        'format': video_id,  # Use specific video format ID
        'outtmpl': f'{save_path}/{yt_id}_video.%(ext)s',
    }
    
    audio_opts = {
        'format': audio_id,  # Use specific audio format ID
        'outtmpl': f'{save_path}/{yt_id}_audio.%(ext)s',
    }

    with yt_dlp.YoutubeDL(video_opts) as ydl:
        ydl.download([url])
    
    with yt_dlp.YoutubeDL(audio_opts) as ydl:
        ydl.download([url])
    
    video_file = f'{save_path}/{yt_id}_video.mp4'  # Assuming mp4 for video
    audio_file = f'{save_path}/{yt_id}_audio.m4a'  # Assuming m4a for audio

    # Merge the video and audio files using ffmpeg
    output_file = f'{save_path}/{yt_id}.mp4'
    merge_command = ['ffmpeg', '-i', video_file, '-i', audio_file, '-c', 'copy', output_file]
    subprocess.run(merge_command, check=True)

    # Clean up intermediate files
    if os.path.exists(video_file):
        os.remove(video_file)
    if os.path.exists(audio_file):
        os.remove(audio_file)

    print(f"Download and merge completed! Output file saved to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python download_video.py <youtube_video_id>")
        sys.exit(1)

    yt_id = sys.argv[1]
    download_best_video_audio(yt_id)
