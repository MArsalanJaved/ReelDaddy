# InstaReels-Creator

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

`YT-Shorts-Maker` is a Python toolset designed to simplify the process of creating engaging Instagram Reels and YouTube Shorts. This tool allows you to download YouTube videos and automatically generate short-form video content that focuses on key moments, such as when someone is speaking. It's perfect for content creators who want to increase their social media engagement effortlessly.

## How It Works

### Step 1: Download a YouTube Video

Use the `downloadYT.py` script to download a video from YouTube.

```bash
python downloadYT.py <youtube-id>
```

- **youtube-id**: This is the unique identifier found in the YouTube video URL. For example, in the URL `https://www.youtube.com/watch?v=dQw4w9WgXcQ`, the `youtube-id` would be `dQw4w9WgXcQ`.

### Step 2: Generate Engaging Reels

Once you have downloaded the video, use the `shortMaker.py` script to generate a reel.

```bash
python shortMaker.py <youtube-id>
```

This script will automatically:

- Cut the video into shorter clips.
- Focus on the person speaking in the video.
- Create engaging reels optimized for Instagram and YouTube.
## Sample Output

Below are examples of how the tool transforms a raw video into engaging content.

### Raw Video

[Link to raw video or embed a preview if possible]

[![Watch the raw video](https://img.youtube.com/vi/qpoRO378qRY/maxresdefault.jpg)](https://www.youtube.com/watch?v=qpoRO378qRY&t=15s)

### Generated Reels Snapshots

Here are a few snapshots of the reels generated by the `shortMaker.py` script:

1. **Snapshot 1**: [Watch the video on Twitter](https://twitter.com/i/status/1819930564264144903)

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/YT-Shorts-Maker.git
cd YT-Shorts-Maker
pip install -r requirements.txt
```

## Dependencies

- `pytube`: For downloading YouTube videos.
- `moviepy`: For editing and processing videos.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you would like to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

This `README.md` file includes all the necessary sections to provide a clear understanding of your project and how to use it. Make sure to replace placeholders like `[Link to raw video or embed a preview if possible]` and `path/to/snapshot1.png` with the actual paths or links in your repository.
